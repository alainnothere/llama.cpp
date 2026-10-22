#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <map>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_DECISION,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    common_chat_input generated_input; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;
    // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
    int32_t n_tokens_shared = 0;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
      server_tokens tokens;

      // Hash of system + first user message tokens.
      // Computed at task creation, used for deterministic cache file naming.
      std::string conversation_id;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_DECISION
    // where to read the model output of each option, exactly one of the two lists is used
    struct decision {
        std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
        std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
        std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
        int32_t                  column = 0;
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        int32_t                  pointer = -1;

        // first prompt position that is read, -1 if none
        int32_t pos_first() const {
            int32_t pos = pointer;
            for (const int32_t marker : markers) {
                pos = pos < 0 ? marker : std::min(pos, marker);
            }
            return pos;
        }

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        std::vector<int32_t> order;
        int32_t              n_scores = 0;
    };
    decision decision;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.markers.empty() || !decision.order.empty();
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.labels.empty();
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }

    // embedding and rerank tasks always process their whole prompt, their state is never reused
    bool is_stateless() const {
        return type == SERVER_TASK_TYPE_EMBEDDING || type == SERVER_TASK_TYPE_RERANK;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    common_chat_input content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    common_chat_input content;
    llama_tokens      tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_decision : server_task_result {
    std::vector<float> scores; // one raw model output per option

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

// FNV-1a 64-bit hash over the token ids [0, tokens_end).
inline uint64_t hash_token_prefix(const server_tokens & tokens, size_t tokens_end) {
    size_t count = std::min(tokens_end, tokens.size());
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t fnv_prime = 0x100000001b3ULL;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&tokens[i]);
        for (size_t b = 0; b < sizeof(llama_token); ++b) {
            hash ^= bytes[b];
            hash *= fnv_prime;
        }
    }
    return hash;
}

// FNV-1a 64-bit hash, returns lowercase hex string.
// Used for checkpoint file naming.
inline std::string generate_conversation_id(const server_tokens & tokens, size_t tokens_end) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) hash_token_prefix(tokens, tokens_end));
    return std::string(buf);
}

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    // Conversation ID: hash of system + first user message tokens.
    // Used for deterministic cache file naming (conversation_id.bin).
    std::string conversation_id;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            conversation_id,
        };
    }
};

// [TAG_REC_REUSE] the hybrid .rec snapshot taken by a slot's last flush_from_context(): the PARTIAL_ONLY
// (recurrent-only) state of ctx_tgt for the slot's sequence, valid at n_tokens. create_checkpoint()
// (server-context.cpp) publishes it as the next checkpoint's data_tgt instead of reading the same 157 MB
// back from the GPU again, when nothing touched the sequence in between (server_slot::state_gen).
struct server_rec_snapshot {
    common_state_data blob;
    int64_t   n_tokens          = -1;
    uint64_t  token_prefix_hash = 0; // hash_token_prefix(tokens, n_tokens)
    llama_pos pos_min           = -1;
    llama_pos pos_max           = -1;
    uint64_t  state_gen         = 0; // server_slot::state_gen at capture, set by the slot

    void clear() {
        blob.clear();
        n_tokens = -1;
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens,
                        std::string disk_path, int32_t queue_depth, int32_t checkpoint_spill_max,
                        uint32_t arch_hash, uint32_t vocab_hash, uint32_t state_hash = 0);
    ~server_prompt_cache();

    server_prompt_cache(const server_prompt_cache &)             = delete;
    server_prompt_cache & operator=(const server_prompt_cache &) = delete;

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // disk tier config — empty disk_path disables the tier
    std::string disk_path;
    int32_t     queue_depth          = 16;
    int32_t     checkpoint_spill_max = -1; // -1 = no limit
    uint32_t    arch_hash            = 0;
    uint32_t    vocab_hash           = 0;
    // KV state identity: cache types + model geometry. Two models can share arch
    // string and vocab (e.g. two sizes of the same family) yet produce mutually
    // unreadable state blobs; this is what tells them apart.
    uint32_t    state_hash           = 0;

    size_t size();
    size_t n_tokens();

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::string & conversation_id, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();

    // called once on the main thread post start_loop() during graceful shutdown.
    // drains the async writer queue (everything scheduled reaches disk), then joins the worker.
    void shutdown_and_spill();

    // write-through: capture the not-yet-flushed tail of the slot's live state as pos-range
    // segments and enqueue them for append to the conversation's .kv/.drft files. models whose
    // serialized state is windowed (SWA / hybrid / recurrent) get a single full-state segment
    // rewrite instead. called from server_slot::prompt_save() while the sequence is still live.
    // pool: where the .rec snapshot buffer comes from (the slot's pool, [TAG_STATE_BLOB]).
    // rec_out: receives a shared reference to the .rec snapshot when one is captured ([TAG_REC_REUSE]);
    // untouched when the flush is a no-op ("already flushed") or the model is not hybrid.
    bool flush_from_context(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                            common_state_pool * pool = nullptr, server_rec_snapshot * rec_out = nullptr);

    // write a single checkpoint to its own cp_ file (async). called from create_checkpoint()
    // in server-context.cpp the moment the checkpoint is captured. no-op if disk_path is empty.
    void spill_checkpoint(const common_prompt_checkpoint & cp, const std::string & conversation_id);

    // convert older resident checkpoints whose cp_ file has committed to disk-backed (lazy)
    // form, keeping only the newest checkpoint's data (and speculative state) in RAM.
    void lazify_checkpoints(server_prompt & prompt);

    // block until the async writer has drained everything currently queued
    void wait_idle();

    // register the prompt's conversation's checkpoint spill files that fill gaps in prompt.checkpoints,
    // from the in-memory cp_ index ([TAG_CP_INDEX]; one directory scan per conversation per process).  called after every successful
    // RAM or disk cache match in load(), and also directly from get_available_slot() when
    // a slot is reused without a cache save/load cycle (f_keep >= 0.5).
    void merge_checkpoint_spills(server_prompt & prompt);

    // forget one cp_ file of the conversation in the in-memory index ([TAG_CP_INDEX]) - called when a
    // disk-backed checkpoint failed to load (the file vanished under us, e.g. the external cache sweep)
    void forget_checkpoint_spill(const std::string & conversation_id, int32_t pos_min);

    // delete this conversation's checkpoint spill files that are bound to more than n_valid
    // tokens. after a rollback those files sit on an abandoned branch: merge always rejects
    // them (see hash_token_prefix), but they still consume the spill budget. called from
    // update_slots() when the new task diverges from the slot's cached tokens.
    void drop_stale_checkpoint_spills(const server_prompt & prompt, int64_t n_valid);

    // true if the host-RAM cache currently exceeds its byte budget (limit_size > 0
    // and size() > limit_size). gates the spill-before-load "valley" path.
    bool over_budget();

    // synchronously write every spillable resident state out to the disk tier and
    // drop it from `states`, freeing host RAM. used before loading a new
    // conversation when RAM-constrained, so the outgoing and incoming states are
    // not both fully resident at once (the spike that can OOM the server when
    // switching between large chats). mtmd entries (not spillable) are dropped.
    // synchronous on the caller's thread; no-op if the disk tier is disabled.
    void spill_all_to_disk();

private:
    // async writer job: either one KV segment append/rewrite or one checkpoint write
    struct disk_job {
        std::string conv_id;

        // segment flush: self-contained state blob covering pos [p0, p1)
        llama_pos p0 = 0;
        llama_pos p1 = 0;
        common_state_data kv;   // [TAG_STATE_BLOB] no zero fill before the range readback
        common_state_data drft;
        std::vector<llama_token> tokens; // tokens covered once this job commits ([0, p1))
        bool rewrite = false;            // first segment of a from-scratch write: truncate the files

        // first segment of a truncate-then-append flush: drop the committed segments above this
        // position (it is always a committed segment end) before appending. -1 = no truncate.
        int32_t truncate_to = -1;

        // hybrid models only: recurrent-state snapshot (PARTIAL_ONLY blob, valid at rec_n_tok),
        // carried by the last segment of a flush and written to the .rec file as [u32 n_tok][blob]
        // (overwrite). shared with the slot's server_rec_snapshot ([TAG_REC_REUSE]), never copied.
        common_state_data rec;
        uint32_t          rec_n_tok = 0;

        // checkpoint write (when set, the segment fields are unused)
        std::shared_ptr<common_prompt_checkpoint> ckpt;

        size_t bytes() const {
            return kv.size() + drft.size() + rec.size() + tokens.size() * sizeof(llama_token) +
                   (ckpt ? ckpt->data_tgt.size() + ckpt->data_dft.size() : 0);
        }
    };

public:
    // per-conversation mirror of the on-disk .hdr plus what has been scheduled into the queue
    // (public: tests hand-craft .hdr files and validate them through read_disk_state)
    struct disk_conv_state {
        struct seg_entry {
            uint64_t kv_off    = 0;
            uint64_t kv_size   = 0;
            uint64_t drft_off  = 0;
            uint64_t drft_size = 0;
            int32_t  p0        = 0;
            int32_t  p1        = 0;
        };

        // committed by the writer thread - matches the files on disk
        uint32_t n_tok     = 0;
        uint64_t kv_size   = 0;
        uint64_t drft_size = 0;
        uint64_t rec_size  = 0; // size of the .rec file; 0 = non-hybrid (no recurrent state)
        std::vector<seg_entry> segments;

        // scheduled by the main thread - committed state plus everything still in the queue
        uint32_t scheduled_n_tok = 0;
        std::vector<llama_token> scheduled_tokens;
        std::vector<uint32_t>    scheduled_seg_ends; // p1 of every scheduled segment, ascending

        // a write failed or the files are inconsistent: the next flush must rewrite from scratch
        bool broken = false;
    };

    // [TAG_CP_INDEX] one cp_{conv}_{pos_min}.bin file as merge_checkpoint_spills() needs it: the header
    // fields plus the blob offsets/sizes, so registering a disk-backed checkpoint opens nothing
    struct cp_file_entry {
        enum status_t : uint8_t {
            CP_OK,        // readable by this build
            CP_BAD_MAGIC, // wrong magic/version or a short header: dead weight, drop_stale removes it
            CP_BAD_HASH,  // another model / cache type wrote it: skipped by merge, kept by drop_stale
            CP_TRUNCATED, // header ok, blobs cut short: skipped by merge
        };

        std::string path;
        int32_t     pos_min           = 0;
        int32_t     pos_max           = 0;
        int64_t     n_tokens          = 0;
        uint64_t    token_prefix_hash = 0;
        uint64_t    off_tgt           = 0;
        uint64_t    sz_tgt            = 0;
        uint64_t    off_dft           = 0;
        uint64_t    sz_dft            = 0;
        status_t    status            = CP_OK;
    };

private:
    // [TAG_CP_INDEX] per conversation: every cp_ file the server knows of, keyed by pos_min (the file
    // name). seeded by one directory scan the first time a conversation is touched in this process
    // (cp_index_seed), then maintained by the writer: added after the rename commits a file, removed when
    // the spill limit evicts one or drop_stale_checkpoint_spills deletes one. files deleted from outside
    // (the operator's sweep cron, every 4 h on grtFxMax) stay listed until a load fails on them: load_tgt
    // returns false, update_slots falls back to an older checkpoint and calls forget_checkpoint_spill().
    // that is the same fallback a vanished file always took. measured 2026-10-22 (27B, 919 entries in
    // the cache dir, ~50 cp_ files per conversation): the scan + 50 opens + a 150 MB seek per file to read
    // an 8-byte draft size ran on every turn inside merge_checkpoint_spills, 10-100 ms.
    struct cp_conv_index {
        bool seeded = false;
        std::map<int32_t, cp_file_entry> files;
    };
    std::unordered_map<std::string, cp_conv_index> cp_index; // guarded by mtx

    // scan disk_path once for the conversation's cp_ files and merge them into cp_index (entries already
    // there win: the writer knows its own files exactly). the scan runs without the mutex
    void cp_index_seed(const std::string & conversation_id);

public:
    // read the header of one cp_ file into an index entry (sizes from the header and the file size)
    bool read_checkpoint_file_entry(const std::string & path, cp_file_entry & e) const;

private:
    std::deque<disk_job>    queue;
    std::mutex              mtx;     // protects queue, queue_bytes, disk_convs, writer_active_conv, cp_index
    std::condition_variable cv;
    std::thread             worker;
    std::atomic<bool>       stop_flag{false};

    std::unordered_map<std::string, disk_conv_state> disk_convs;

    size_t      queue_bytes = 0;        // payload bytes currently sitting in `queue`
    std::string writer_active_conv;     // conv_id the writer is persisting right now

    // launched lazily once disk_path is non-empty
    void start_writer_if_needed();
    void writer_loop();

    // file I/O -- runs on the worker thread
    void process_disk_job(const disk_job & job);
    bool write_checkpoint_file(const common_prompt_checkpoint & cp, const std::string & conversation_id);

public:
    // seed a disk_conv_state's committed fields from an existing .hdr (server restart case)
    bool read_disk_state(const std::string & conversation_id, disk_conv_state & st);

    // direct disk load by conversation_id: reads .hdr and .tok, then streams the .kv/.drft
    // segments into the context one at a time (bounded RAM - peak is one segment)
    bool load_from_disk(server_prompt & prompt, const std::string & conversation_id,
                        llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
