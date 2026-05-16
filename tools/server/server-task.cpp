#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <system_error>

using json = nlohmann::ordered_json;

//
// disk-cache helpers (file-local)
//

namespace {

constexpr uint32_t DISK_CACHE_MAGIC   = 0x53504344; // 'SPCD' — Server Prompt Cache Disk
// v2: appended per-entry checkpoints (n_tokens, pos_min, pos_max, data_tgt, data_dft) after the
// main+drft blobs. Required for SWA / hybrid / recurrent models: PARTIAL_ONLY checkpoints are the
// only way to reuse a prefix once the live KV window has slid past it. v1 files are auto-deleted
// by header-mismatch handling in try_match_disk.
constexpr uint32_t DISK_CACHE_VERSION = 2;

// checkpoint spill files — written by spill_checkpoint() when create_checkpoint() evicts an old
// checkpoint to make room for a new one. format: magic, version, arch_hash, vocab_hash,
// pos_min (i32), pos_max (i32), n_tokens (i64), data_tgt_size (u64), data_tgt, data_dft_size (u64), data_dft.
// files are named cp_{pos_min}_{uuid}.bin and live in the same disk_path as cache entry files.
// try_match_disk skips them (wrong magic); merge_checkpoint_spills() reads them.
constexpr uint32_t CHECKPOINT_SPILL_MAGIC   = 0x43504B44; // 'CPKD'
constexpr uint32_t CHECKPOINT_SPILL_VERSION = 1;

std::string gen_disk_cache_uuid() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t a = dist(rng);
    const uint64_t b = dist(rng);
    char buf[33];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                  (unsigned long long) a, (unsigned long long) b);
    return std::string(buf);
}

} // anonymous namespace

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// server_task
//

task_params server_task::params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
    task_params defaults;
    defaults.sampling      = params_base.sampling;
    defaults.speculative   = params_base.speculative;
    defaults.n_keep        = params_base.n_keep;
    defaults.n_predict     = params_base.n_predict;
    defaults.n_cache_reuse = params_base.n_cache_reuse;
    defaults.cache_prompt  = params_base.cache_prompt;
    defaults.antiprompt    = params_base.antiprompt;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose           = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream           = json_value(data,       "stream",             false);
    auto stream_opt         = json_value(data,       "stream_options",     json::object());
    params.include_usage    = json_value(stream_opt, "include_usage",      false);
    params.cache_prompt     = json_value(data,       "cache_prompt",       defaults.cache_prompt);
    params.return_tokens    = json_value(data,       "return_tokens",      false);
    params.return_progress  = json_value(data,       "return_progress",    false);
    auto max_tokens         = json_value(data,       "max_tokens",         defaults.n_predict);
    params.n_predict        = json_value(data,       "n_predict",          json_value(data, "max_completion_tokens", max_tokens));
    params.n_indent         = json_value(data,       "n_indent",           defaults.n_indent);
    params.n_keep           = json_value(data,       "n_keep",             defaults.n_keep);
    params.n_discard        = json_value(data,       "n_discard",          defaults.n_discard);
    params.n_discard        = std::max(0, params.n_discard);
    params.n_cmpl           = json_value(data,       "n_cmpl",             json_value(data, "n", 1));
    params.n_cache_reuse    = json_value(data,       "n_cache_reuse",      defaults.n_cache_reuse);
    //params.t_max_prompt_ms  = json_value(data,       "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data,       "t_max_predict_ms",   defaults.t_max_predict_ms);
    params.response_fields  = json_value(data,       "response_fields",    std::vector<std::string>());

    params.sampling.top_k              = json_value(data, "top_k",               defaults.sampling.top_k);
    params.sampling.top_p              = json_value(data, "top_p",               defaults.sampling.top_p);
    params.sampling.min_p              = json_value(data, "min_p",               defaults.sampling.min_p);
    params.sampling.top_n_sigma        = json_value(data, "top_n_sigma",         defaults.sampling.top_n_sigma);
    params.sampling.xtc_probability    = json_value(data, "xtc_probability",     defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",       defaults.sampling.xtc_threshold);
    params.sampling.typ_p              = json_value(data, "typical_p",           defaults.sampling.typ_p);
    params.sampling.temp               = json_value(data, "temperature",         defaults.sampling.temp);
    params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",      defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",   defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",       defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",      defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq       = json_value(data, "frequency_penalty",   defaults.sampling.penalty_freq);
    params.sampling.penalty_present    = json_value(data, "presence_penalty",    defaults.sampling.penalty_present);
    params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",      defaults.sampling.dry_multiplier);
    params.sampling.dry_base           = json_value(data, "dry_base",            defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length",  defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n",  defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat           = json_value(data, "mirostat",            defaults.sampling.mirostat);
    params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",        defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",        defaults.sampling.mirostat_eta);
    params.sampling.adaptive_target    = json_value(data, "adaptive_target",     defaults.sampling.adaptive_target);
    params.sampling.adaptive_decay     = json_value(data, "adaptive_decay",      defaults.sampling.adaptive_decay);
    params.sampling.seed               = json_value(data, "seed",                defaults.sampling.seed);
    params.sampling.n_probs            = json_value(data, "n_probs",             defaults.sampling.n_probs);
    params.sampling.min_keep           = json_value(data, "min_keep",            defaults.sampling.min_keep);
    params.sampling.backend_sampling   = json_value(data, "backend_sampling",    defaults.sampling.backend_sampling);
    params.post_sampling_probs         = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative = defaults.speculative;

    // TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    params.speculative.draft.n_min = json_value(data, "speculative.n_min", defaults.speculative.draft.n_min);
    params.speculative.draft.n_max = json_value(data, "speculative.n_max", defaults.speculative.draft.n_max);
    params.speculative.draft.p_min = json_value(data, "speculative.p_min", defaults.speculative.draft.p_min);

    params.speculative.draft.n_min = std::min(params.speculative.draft.n_max, params.speculative.draft.n_min);
    params.speculative.draft.n_min = std::max(params.speculative.draft.n_min, 0);
    params.speculative.draft.n_max = std::max(params.speculative.draft.n_max, 0);

    // for debugging and research purposes
    params.speculative.type = common_speculative_type_from_name(json_value(data, "speculative.type", common_speculative_type_to_str(defaults.speculative.type)));

    params.speculative.ngram_size_n     = json_value(data, "speculative.ngram_size_n", defaults.speculative.ngram_size_n);
    params.speculative.ngram_size_m     = json_value(data, "speculative.ngram_size_m", defaults.speculative.ngram_size_m);
    params.speculative.ngram_min_hits   = json_value(data, "speculative.ngram_m_hits", defaults.speculative.ngram_min_hits);

    params.speculative.ngram_size_n     = std::max(std::min(1, (int) params.speculative.ngram_size_n),     1024);
    params.speculative.ngram_size_m     = std::max(std::min(1, (int) params.speculative.ngram_size_m),     1024);
    params.speculative.ngram_min_hits   = std::max(std::min(1, (int) params.speculative.ngram_min_hits),   1024);
#endif

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
        params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
        if (data.at("lora").is_array()) {
            params.lora = parse_lora_request(data.at("lora"));
        } else {
            throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
        }
    } else {
        params.lora = {};
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
        // note: should be the slot's context and not the full context, but it's ok
        params.sampling.penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        params.sampling.dry_penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_base < 1.0f) {
        params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema                  = json_value(data, "json_schema", json::object());
            SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
            std::string grammar_str      = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
            params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    } else {
        params.sampling.grammar = defaults.sampling.grammar;

        std::string grammar_str = json_value(data, "grammar", std::string());
        if (!grammar_str.empty()) {
            // grammar_type key is set by the server when converting chat template grammars
            std::string grammar_type = json_value(data, "grammar_type", std::string());
            if (grammar_type == "tool_calls") {
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
            } else {
                // explicit grammar from the user (API field "grammar")
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
            }
            SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
        }
        params.sampling.grammar_lazy = json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
        SRV_DBG("Grammar lazy: %s\n", params.sampling.grammar_lazy ? "true" : "false");
    }

    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.chat_parser_params.format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.chat_parser_params.format));
        } else {
            params.chat_parser_params.format = defaults.chat_parser_params.format;
        }
        common_reasoning_format reasoning_format = params_base.reasoning_format;
        if (data.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }
        params.chat_parser_params.reasoning_format = reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        params.chat_parser_params.generation_prompt = json_value(data, "generation_prompt", std::string());
        params.sampling.generation_prompt = params.chat_parser_params.generation_prompt;
        SRV_DBG("Generation prompt: '%s'\n", params.chat_parser_params.generation_prompt.c_str());
        params.chat_parser_params.parse_tool_calls = json_value(data, "parse_tool_calls", false);
        if (data.contains("chat_parser")) {
            params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }
    }

    {
        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            for (const auto & t : *preserved_tokens) {
                auto ids = common_tokenize(vocab, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    SRV_DBG("Preserved token: %d\n", ids[0]);
                    params.sampling.preserved_tokens.insert(ids[0]);
                } else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    SRV_DBG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            for (const auto & t : *grammar_triggers) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value.value;
                    auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                        params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        SRV_DBG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                    } else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                        SRV_DBG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                    } else {
                        throw std::runtime_error("Unknown grammar trigger type");
                    }
                    params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
        }
        if (params.sampling.grammar_lazy && params.sampling.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    // Parse reasoning budget sampler parameters
    {
        const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t) -1);
        const auto start_tag = json_value(data, "reasoning_budget_start_tag", std::string());
        const auto end_tag   = json_value(data, "reasoning_budget_end_tag", std::string());
        const auto message   = json_value(data, "reasoning_budget_message", std::string());
        params.sampling.reasoning_budget_tokens = budget;

        if (!start_tag.empty()) {
            params.sampling.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (!end_tag.empty()) {
            params.sampling.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            params.sampling.reasoning_budget_forced = common_tokenize(vocab, message + end_tag, false, true);

            SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
        }
    }

    {
        params.sampling.logit_bias.clear();

        const auto & logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    } else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    } else if (el[0].is_string()) {
                        auto toks = common_tokenize(vocab, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    }
                }
            }
        } else if (logit_bias != data.end() && logit_bias->is_object()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : logit_bias->items()) {
                float bias;
                const auto & key = el.key();
                const auto & value = el.value();
                if (value.is_number()) {
                    bias = value.get<float>();
                } else if (value.is_boolean() && !value.get<bool>()) {
                    bias = -INFINITY;
                } else {
                    continue;
                }

                char *end;
                llama_token tok = strtol(key.c_str(), &end, 10);
                if (*end == 0) {
                    if (tok >= 0 && tok < n_vocab) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                } else {
                    auto toks = common_tokenize(vocab, key, false);
                    for (auto tok : toks) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            }
        }

        params.sampling.ignore_eos = json_value(data, "ignore_eos", params_base.sampling.ignore_eos);
        if (params.sampling.ignore_eos) {
            params.sampling.logit_bias.insert(
                    params.sampling.logit_bias.end(),
                    logit_bias_eog.begin(), logit_bias_eog.end());
        }
    }

    {
        params.antiprompt.clear();

        const auto & stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto & word : *stop) {
                if (!word.empty()) {
                    params.antiprompt.push_back(word);
                }
            }
        }
        // set reverse prompt from cli args if not set in the request
        if (params.antiprompt.empty()) {
            params.antiprompt = defaults.antiprompt;
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                params.sampling.samplers = common_sampler_types_from_names(*samplers, false);
            } else if (samplers->is_string()){
                params.sampling.samplers = common_sampler_types_from_chars(samplers->get<std::string>());
            }
        } else {
            params.sampling.samplers = defaults.sampling.samplers;
        }
    }

    if (params.n_cmpl > params_base.n_parallel) {
        throw std::runtime_error("n_cmpl cannot be greater than the number of slots, please increase -np");
    }

    return params;
}

//
// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }
    // pending_spill is shared with the writer thread — lock briefly to walk it safely
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto & ptr : pending_spill) {
            res += ptr->size();
        }
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto & ptr : pending_spill) {
            res += ptr->n_tokens();
        }
    }

    return res;
}

server_prompt * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_INF("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // next, remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->tokens.size()) {
            SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.checkpoints =*/ prompt.checkpoints,
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = float(lcp_best) / tokens_new.size();

    SRV_INF(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best = states.end();

    // Phase 1: find best match in states
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float sim_cur    = float(lcp_cur) / tokens_new.size();

        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;

            it_best = it;
        }
    }

    // Phase 2: scan pending_spill (locked). If a pending entry beats the states winner, consume it from pending.
    std::shared_ptr<server_prompt> pending_winner;
    {
        std::unique_lock<std::mutex> lock(mtx);
        auto best = pending_spill.end();
        float pf_keep = f_keep_best;
        float psim    = sim_best;
        for (auto it = pending_spill.begin(); it != pending_spill.end(); ++it) {
            const int lcp_cur = (*it)->tokens.get_common_prefix(tokens_new);
            const float f_keep_cur = float(lcp_cur) / (*it)->tokens.size();
            const float sim_cur    = float(lcp_cur) / tokens_new.size();
            if (f_keep_cur < 0.25f) {
                continue;
            }
            if (pf_keep < f_keep_cur && psim < sim_cur) {
                pf_keep = f_keep_cur;
                psim    = sim_cur;
                best = it;
            }
        }
        if (best != pending_spill.end()) {
            pending_winner = *best;
            pending_spill.erase(best);
            f_keep_best = pf_keep;
            sim_best    = psim;
            it_best     = states.end(); // pending wins over states
        }
    }

    if (pending_winner) {
        SRV_WRN(" - found better prompt in pending_spill with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        {
            auto & data = pending_winner->data.main;
            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_WRN("failed to restore state with size %zu\n", size);
                return false;
            }
            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = pending_winner->data.drft;
            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);
                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore drft state with size %zu\n", size);
                    return false;
                }
                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(*pending_winner);
        if (!disk_path.empty()) {
            merge_checkpoint_spills(prompt);
        }
        return true;
    }

    if (it_best != states.end()) {
        SRV_INF(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(*it_best);

        states.erase(it_best);
        if (!disk_path.empty()) {
            merge_checkpoint_spills(prompt);
        }
        return true;
    }

    // Phase 3: disk fallback
    if (!disk_path.empty()) {
        if (try_match_disk(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot)) {
            SRV_WRN("%s", " - restored from disk\n");
            merge_checkpoint_spills(prompt);
            return true;
        }
    }

    return true;
}

void server_prompt_cache::update() {
    auto evict_oldest = [this]() {
        // Either spill to disk (preserving the entry) or destroy it.
        // mtmd entries cannot be spilled — they get destroyed either way.
        if (states.empty()) return;

        const bool has_mtmd = states.front().tokens.has_mtmd;
        if (disk_path.empty() || has_mtmd) {
            SRV_WRN(" - removing oldest entry (size = %.3f MiB)%s\n",
                    states.front().size() / (1024.0 * 1024.0),
                    (!disk_path.empty() && has_mtmd) ? " [mtmd: not spillable]" : "");
            states.pop_front();
            return;
        }

        // Spill: move oldest entry to pending_spill, enqueue for async write.
        // On queue overflow, fall through to a synchronous write on this thread (lossless).
        std::string uuid     = gen_disk_cache_uuid();
        std::string filepath = disk_path + "/" + uuid + ".bin";

        auto entry_ptr = std::make_shared<server_prompt>(std::move(states.front()));
        states.pop_front();

        std::unique_lock<std::mutex> lock(mtx);
        pending_spill.push_back(entry_ptr);

        if ((int32_t) queue.size() < queue_depth) {
            queue.push_back({uuid, filepath, entry_ptr});
            cv.notify_one();
            SRV_WRN(" - spilling oldest entry to disk (async, queued %zu/%d, size = %.3f MiB)\n",
                    queue.size(), queue_depth, entry_ptr->size() / (1024.0 * 1024.0));
        } else {
            // Queue full — write synchronously (the task thread pays the cost; never drop)
            lock.unlock();
            SRV_WRN(" - spilling oldest entry to disk (SYNC fall-through, size = %.3f MiB)\n",
                    entry_ptr->size() / (1024.0 * 1024.0));
            const bool ok = write_entry_to_file(filepath, *entry_ptr);
            lock.lock();
            for (auto it = pending_spill.begin(); it != pending_spill.end(); ++it) {
                if (it->get() == entry_ptr.get()) {
                    pending_spill.erase(it);
                    break;
                }
            }
            if (!ok) {
                SRV_WRN("disk cache: synchronous spill failed for '%s' (entry dropped)\n", filepath.c_str());
            }
        }
    };

    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            evict_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            evict_oldest();
        }
    }

    SRV_INF(" - cache state: %zu prompts, (+%zu pending), %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), pending_spill.size(), size() / (1024.0 * 1024.0),
            limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_INF("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_prompt_cache — disk tier
//

server_prompt_cache::server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens_in,
                                          std::string disk_path_in, int32_t queue_depth_in,
                                          uint32_t arch_hash_in, uint32_t vocab_hash_in) {
    this->limit_size   = 1024ull * 1024ull * (limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens_in;
    this->disk_path    = std::move(disk_path_in);
    this->queue_depth  = queue_depth_in > 0 ? queue_depth_in : 16;
    this->arch_hash    = arch_hash_in;
    this->vocab_hash   = vocab_hash_in;

    if (!this->disk_path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(this->disk_path, ec);
        if (ec) {
            SRV_WRN("disk cache: cannot create directory '%s': %s — disk tier disabled\n",
                    this->disk_path.c_str(), ec.message().c_str());
            this->disk_path.clear();
        } else {
            SRV_WRN("disk cache: enabled at '%s' (queue depth %d, arch=0x%08x, vocab=0x%08x)\n",
                    this->disk_path.c_str(), this->queue_depth, this->arch_hash, this->vocab_hash);
            start_writer_if_needed();
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    // Destructor runs on the main thread when ctx_server is destroyed (post start_loop, post clean_up).
    // shutdown_and_spill stops the writer, joins, then synchronously spills remaining states + pending entries.
    // The signal handler never reaches this path (it only sets stop_flag indirectly via terminate()).
    shutdown_and_spill();
}

void server_prompt_cache::start_writer_if_needed() {
    if (worker.joinable() || disk_path.empty()) {
        return;
    }
    worker = std::thread([this]() { this->writer_loop(); });
}

void server_prompt_cache::writer_loop() {
    while (true) {
        spill_job job;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return stop_flag.load() || !queue.empty(); });
            if (stop_flag.load() && queue.empty()) {
                return;
            }
            job = std::move(queue.front());
            queue.pop_front();
        }

        // I/O outside the lock — writes can take seconds
        const bool ok = write_entry_to_file(job.filepath, *job.entry);
        if (!ok) {
            SRV_WRN("disk cache: async write failed for '%s' (entry dropped)\n", job.filepath.c_str());
        }

        // On failure: drop the entry from pending_spill rather than re-queuing it for retry.
        //
        // Why drop instead of retry:
        // - Real-world write failures here are dominated by persistent causes (ENOSPC, EACCES,
        //   EIO from failing hardware, path unmounted). Retry hits the same wall.
        // - "Move back to states for retry" turns persistent failure into livelock: the failed
        //   entry returns to states, next update() picks it up, spills, fails, returns... the
        //   loop never terminates, RAM stays maxed, CPU pegged, logs flooded. Re-queue does not
        //   achieve losslessness for persistent failures — it just relocates the loss into
        //   "loop forever, never make progress."
        // - The lossless property this cache actually provides is around queue overflow
        //   (synchronous fall-through, not drop). Write-failure handling is a separate axis.
        // - Drop with a warning gives the operator a clear signal something's wrong while
        //   keeping the cache functional at reduced effective capacity.
        //
        // If transient-failure recovery ever becomes important, the right shape is bounded
        // retry with backoff (e.g., 3 attempts, 100/500/2000 ms), then drop — not unconditional
        // re-queue.
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (auto it = pending_spill.begin(); it != pending_spill.end(); ++it) {
                if (it->get() == job.entry.get()) {
                    pending_spill.erase(it);
                    break;
                }
            }
            cv.notify_all();
        }
    }
}

bool server_prompt_cache::write_entry_to_file(const std::string & filepath, const server_prompt & entry) {
    const std::string tmppath = filepath + ".tmp";

    size_t ckpts_bytes = 0;
    for (const auto & cp : entry.checkpoints) {
        ckpts_bytes += cp.data_tgt.size() + cp.data_dft.size();
    }
    const size_t bytes_total = entry.data.main.size() + entry.data.drft.size() + ckpts_bytes;
    const int    n_tok       = entry.n_tokens();
    const int64_t t_start    = ggml_time_us();

    SRV_WRN("disk cache: writing %d tokens, %.1f MiB (main+drft) + %zu checkpoint(s) (%.1f MiB) → %s\n",
            n_tok,
            (entry.data.main.size() + entry.data.drft.size()) / (1024.0 * 1024.0),
            entry.checkpoints.size(),
            ckpts_bytes / (1024.0 * 1024.0),
            filepath.c_str());

    {
        std::ofstream f(tmppath, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }

        auto put_u32 = [&f](uint32_t v) {
            f.write(reinterpret_cast<const char *>(&v), sizeof(v));
        };
        auto put_u64 = [&f](uint64_t v) {
            f.write(reinterpret_cast<const char *>(&v), sizeof(v));
        };

        put_u32(DISK_CACHE_MAGIC);
        put_u32(DISK_CACHE_VERSION);
        put_u32(arch_hash);
        put_u32(vocab_hash);
        put_u32(static_cast<uint32_t>(limit_tokens)); // n_ctx at spill time (stored for diagnostics)

        const auto & toks = entry.tokens.get_tokens();
        const uint32_t n_tok = static_cast<uint32_t>(toks.size());
        put_u32(n_tok);
        if (n_tok > 0) {
            f.write(reinterpret_cast<const char *>(toks.data()),
                    static_cast<std::streamsize>(n_tok * sizeof(llama_token)));
        }

        put_u64(static_cast<uint64_t>(entry.data.main.size()));
        if (!entry.data.main.empty()) {
            f.write(reinterpret_cast<const char *>(entry.data.main.data()),
                    static_cast<std::streamsize>(entry.data.main.size()));
        }

        put_u64(static_cast<uint64_t>(entry.data.drft.size()));
        if (!entry.data.drft.empty()) {
            f.write(reinterpret_cast<const char *>(entry.data.drft.data()),
                    static_cast<std::streamsize>(entry.data.drft.size()));
        }

        // checkpoints — PARTIAL_ONLY KV snapshots taken during prefill. Required for SWA / hybrid /
        // recurrent models: the main blob captures the trailing window only, so without these the
        // rewind path at server-context.cpp can't recover earlier positions across restart.
        put_u32(static_cast<uint32_t>(entry.checkpoints.size()));
        for (const auto & cp : entry.checkpoints) {
            const int64_t  n_tokens_v = cp.n_tokens;
            const int32_t  pos_min_v  = cp.pos_min;
            const int32_t  pos_max_v  = cp.pos_max;
            f.write(reinterpret_cast<const char *>(&n_tokens_v), sizeof(n_tokens_v));
            f.write(reinterpret_cast<const char *>(&pos_min_v),  sizeof(pos_min_v));
            f.write(reinterpret_cast<const char *>(&pos_max_v),  sizeof(pos_max_v));

            put_u64(static_cast<uint64_t>(cp.data_tgt.size()));
            if (!cp.data_tgt.empty()) {
                f.write(reinterpret_cast<const char *>(cp.data_tgt.data()),
                        static_cast<std::streamsize>(cp.data_tgt.size()));
            }
            put_u64(static_cast<uint64_t>(cp.data_dft.size()));
            if (!cp.data_dft.empty()) {
                f.write(reinterpret_cast<const char *>(cp.data_dft.data()),
                        static_cast<std::streamsize>(cp.data_dft.size()));
            }
        }

        if (!f) {
            std::error_code ec;
            std::filesystem::remove(tmppath, ec);
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmppath, filepath, ec);
    if (ec) {
        std::filesystem::remove(tmppath, ec);
        return false;
    }

    const double t_ms = (ggml_time_us() - t_start) / 1e3;
    const double mb_s = t_ms > 0.0 ? (bytes_total / (1024.0 * 1024.0)) / (t_ms / 1e3) : 0.0;
    SRV_WRN("disk cache: wrote %.1f MiB in %.1f s (%.1f MiB/s) → %s\n",
            bytes_total / (1024.0 * 1024.0), t_ms / 1e3, mb_s, filepath.c_str());
    return true;
}

void server_prompt_cache::spill_checkpoint(const common_prompt_checkpoint & cp) {
    if (disk_path.empty()) {
        return;
    }

    const std::string uuid     = gen_disk_cache_uuid();
    const std::string filename = "cp_" + std::to_string(cp.pos_min) + "_" + uuid + ".bin";
    const std::filesystem::path filepath = std::filesystem::path(disk_path) / filename;
    const std::filesystem::path tmppath  = std::filesystem::path(disk_path) / (filename + ".tmp");

    {
        std::ofstream f(tmppath, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_WRN("checkpoint spill: failed to open '%s' for write\n", tmppath.string().c_str());
            return;
        }

        auto put_u32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
        auto put_u64 = [&f](uint64_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };

        put_u32(CHECKPOINT_SPILL_MAGIC);
        put_u32(CHECKPOINT_SPILL_VERSION);
        put_u32(arch_hash);
        put_u32(vocab_hash);

        const int32_t pos_min_v  = cp.pos_min;
        const int32_t pos_max_v  = cp.pos_max;
        const int64_t n_tokens_v = cp.n_tokens;
        f.write(reinterpret_cast<const char *>(&pos_min_v),  sizeof(pos_min_v));
        f.write(reinterpret_cast<const char *>(&pos_max_v),  sizeof(pos_max_v));
        f.write(reinterpret_cast<const char *>(&n_tokens_v), sizeof(n_tokens_v));

        put_u64(static_cast<uint64_t>(cp.data_tgt.size()));
        if (!cp.data_tgt.empty()) {
            f.write(reinterpret_cast<const char *>(cp.data_tgt.data()),
                    static_cast<std::streamsize>(cp.data_tgt.size()));
        }

        put_u64(static_cast<uint64_t>(cp.data_dft.size()));
        if (!cp.data_dft.empty()) {
            f.write(reinterpret_cast<const char *>(cp.data_dft.data()),
                    static_cast<std::streamsize>(cp.data_dft.size()));
        }

        if (!f) {
            std::error_code ec;
            std::filesystem::remove(tmppath, ec);
            SRV_WRN("checkpoint spill: write error for '%s'\n", tmppath.string().c_str());
            return;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmppath, filepath, ec);
    if (ec) {
        std::filesystem::remove(tmppath, ec);
        SRV_WRN("checkpoint spill: rename failed '%s': %s\n", tmppath.string().c_str(), ec.message().c_str());
        return;
    }

    SRV_WRN("REMOVE_ME checkpoint spill: pos_min=%d pos_max=%d n_tokens=%lld tgt=%.1f MiB dft=%.1f MiB → %s\n",
            cp.pos_min, cp.pos_max, (long long) cp.n_tokens,
            cp.data_tgt.size() / (1024.0 * 1024.0),
            cp.data_dft.size() / (1024.0 * 1024.0),
            filename.c_str());
}

void server_prompt_cache::merge_checkpoint_spills(server_prompt & prompt) {
    if (disk_path.empty()) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::exists(disk_path, ec) || ec) {
        return;
    }

    int n_scanned    = 0;
    int n_loaded     = 0;
    int n_skip_magic = 0;
    int n_skip_hash  = 0;
    int n_skip_dup   = 0;

    std::list<common_prompt_checkpoint> new_ckpts;

    for (const auto & entry : std::filesystem::directory_iterator(disk_path, ec)) {
        if (ec) break;
        const std::string fname = entry.path().filename().string();
        if (fname.size() < 4 || fname.compare(0, 3, "cp_") != 0) {
            continue;
        }
        if (entry.path().extension() != ".bin") {
            continue;
        }
        ++n_scanned;

        std::ifstream f(entry.path(), std::ios::binary);
        if (!f) {
            continue;
        }

        uint32_t magic = 0, version = 0, fa = 0, fv = 0;
        f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (magic != CHECKPOINT_SPILL_MAGIC || version != CHECKPOINT_SPILL_VERSION) {
            ++n_skip_magic;
            continue;
        }

        f.read(reinterpret_cast<char *>(&fa), sizeof(fa));
        f.read(reinterpret_cast<char *>(&fv), sizeof(fv));

        if (fa != arch_hash || fv != vocab_hash) {
            ++n_skip_hash;
            continue;
        }

        int32_t pos_min_v  = 0;
        int32_t pos_max_v  = 0;
        int64_t n_tokens_v = 0;
        f.read(reinterpret_cast<char *>(&pos_min_v),  sizeof(pos_min_v));
        f.read(reinterpret_cast<char *>(&pos_max_v),  sizeof(pos_max_v));
        f.read(reinterpret_cast<char *>(&n_tokens_v), sizeof(n_tokens_v));
        if (!f) {
            continue;
        }

        // skip if this pos_min is already in the checkpoint list
        bool dup = false;
        for (const auto & cp : prompt.checkpoints) {
            if (cp.pos_min == pos_min_v) {
                dup = true;
                break;
            }
        }
        if (dup) {
            ++n_skip_dup;
            SRV_WRN("REMOVE_ME merge_checkpoint_spills: skip dup pos_min=%d from '%s'\n", pos_min_v, fname.c_str());
            continue;
        }

        uint64_t tgt_sz = 0;
        f.read(reinterpret_cast<char *>(&tgt_sz), sizeof(tgt_sz));
        if (!f) {
            continue;
        }

        common_prompt_checkpoint cp;
        cp.pos_min  = pos_min_v;
        cp.pos_max  = pos_max_v;
        cp.n_tokens = n_tokens_v;

        if (tgt_sz > 0) {
            cp.data_tgt.resize(static_cast<size_t>(tgt_sz));
            f.read(reinterpret_cast<char *>(cp.data_tgt.data()),
                   static_cast<std::streamsize>(tgt_sz));
            if (!f) {
                continue;
            }
        }

        uint64_t dft_sz = 0;
        f.read(reinterpret_cast<char *>(&dft_sz), sizeof(dft_sz));
        if (!f) {
            continue;
        }
        if (dft_sz > 0) {
            cp.data_dft.resize(static_cast<size_t>(dft_sz));
            f.read(reinterpret_cast<char *>(cp.data_dft.data()),
                   static_cast<std::streamsize>(dft_sz));
            if (!f) {
                continue;
            }
        }

        SRV_WRN("REMOVE_ME merge_checkpoint_spills: loaded pos_min=%d pos_max=%d n_tokens=%lld"
                " tgt=%.1f MiB dft=%.1f MiB from '%s'\n",
                pos_min_v, pos_max_v, (long long) n_tokens_v,
                tgt_sz / (1024.0 * 1024.0), dft_sz / (1024.0 * 1024.0),
                fname.c_str());

        new_ckpts.push_back(std::move(cp));
        ++n_loaded;
    }

    SRV_WRN("REMOVE_ME merge_checkpoint_spills: scanned=%d loaded=%d skip_magic=%d skip_hash=%d skip_dup=%d existing=%zu\n",
            n_scanned, n_loaded, n_skip_magic, n_skip_hash, n_skip_dup, prompt.checkpoints.size());

    if (new_ckpts.empty()) {
        return;
    }

    // insert into prompt.checkpoints keeping sorted order by pos_min
    for (auto & cp : new_ckpts) {
        auto it = prompt.checkpoints.begin();
        while (it != prompt.checkpoints.end() && it->pos_min < cp.pos_min) {
            ++it;
        }
        prompt.checkpoints.insert(it, std::move(cp));
    }

    SRV_WRN("REMOVE_ME merge_checkpoint_spills: after merge prompt.checkpoints.size=%zu\n",
            prompt.checkpoints.size());
}

bool server_prompt_cache::try_match_disk(server_prompt & prompt, const server_tokens & tokens_new,
                                          llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    SRV_WRN("REMOVE_ME try_match_disk: enter, disk_path='%s', tokens_new.size=%zu, id_slot=%d, ctx_dft=%p\n",
            disk_path.c_str(), tokens_new.size(), id_slot, (void *) ctx_dft);
    if (disk_path.empty()) {
        SRV_WRN("%s", "REMOVE_ME try_match_disk: disk_path empty → return false\n");
        return false;
    }

    std::error_code ec;
    if (!std::filesystem::exists(disk_path, ec) || ec) {
        SRV_WRN("REMOVE_ME try_match_disk: disk_path does not exist (ec=%s) → return false\n", ec.message().c_str());
        return false;
    }

    struct disk_match {
        std::filesystem::path path;
        std::vector<llama_token> tokens;
        uint64_t main_size = 0;
        uint64_t drft_size = 0;
        std::streampos main_offset {};
        float f_keep = 0.0f;
        float sim    = 0.0f;
    };

    std::optional<disk_match> best;
    float f_keep_best = 0.25f; // threshold floor
    float sim_best    = -1.0f;

    const bool runtime_has_dft = (ctx_dft != nullptr);

    int n_scanned = 0;
    int n_rejected_header = 0;
    int n_rejected_hash = 0;
    int n_rejected_capacity = 0;
    int n_rejected_asym = 0;
    int n_rejected_score = 0;
    for (const auto & entry : std::filesystem::directory_iterator(disk_path, ec)) {
        if (ec) break;
        if (entry.path().extension() != ".bin") continue;
        ++n_scanned;

        std::ifstream f(entry.path(), std::ios::binary);
        if (!f) continue;

        auto get_u32 = [&f]() -> std::optional<uint32_t> {
            uint32_t v;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (!f) return std::nullopt;
            return v;
        };
        auto get_u64 = [&f]() -> std::optional<uint64_t> {
            uint64_t v;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (!f) return std::nullopt;
            return v;
        };

        const auto magic   = get_u32();
        const auto version = get_u32();

        if (!magic || *magic != DISK_CACHE_MAGIC) {
            // not a cache entry file — skip without deleting (may be a checkpoint spill file)
            SRV_WRN("REMOVE_ME try_match_disk: '%s' wrong magic (0x%08x) — skip\n",
                    entry.path().filename().c_str(), magic ? *magic : 0u);
            continue;
        }

        if (!version || *version != DISK_CACHE_VERSION) {
            ++n_rejected_header;
            SRV_WRN("REMOVE_ME try_match_disk: '%s' wrong version (%s) → delete\n",
                    entry.path().filename().c_str(), version ? std::to_string(*version).c_str() : "?");
            f.close();
            std::filesystem::remove(entry.path(), ec);
            continue;
        }

        const auto fa = get_u32();
        const auto fv = get_u32();
        const auto fn_ctx = get_u32();
        const auto n_tok  = get_u32();
        (void) fn_ctx;
        if (!fa || !fv || !n_tok) { ++n_rejected_header; continue; }
        if (*fa != arch_hash || *fv != vocab_hash) {
            ++n_rejected_hash;
            SRV_WRN("REMOVE_ME try_match_disk: '%s' arch/vocab mismatch (file arch=0x%08x vocab=0x%08x, runtime arch=0x%08x vocab=0x%08x) → skip\n",
                    entry.path().filename().c_str(), *fa, *fv, arch_hash, vocab_hash);
            continue;
        }
        if (limit_tokens > 0 && *n_tok > limit_tokens) {
            ++n_rejected_capacity;
            SRV_WRN("REMOVE_ME try_match_disk: '%s' too many tokens (%u > %zu) → skip\n",
                    entry.path().filename().c_str(), *n_tok, limit_tokens);
            continue;
        }

        std::vector<llama_token> toks(*n_tok);
        if (*n_tok > 0) {
            f.read(reinterpret_cast<char *>(toks.data()),
                   static_cast<std::streamsize>(*n_tok * sizeof(llama_token)));
            if (!f) continue;
        }

        const auto main_size = get_u64();
        if (!main_size) continue;
        const auto main_offset = f.tellg();
        f.seekg(static_cast<std::streamoff>(*main_size), std::ios::cur);
        const auto drft_size = get_u64();
        if (!drft_size) continue;

        // asymmetric file rule: runtime has draft but file has none → skip
        if (runtime_has_dft && *drft_size == 0) {
            ++n_rejected_asym;
            SRV_WRN("REMOVE_ME try_match_disk: '%s' asymmetric (runtime has drft, file does not) → skip\n",
                    entry.path().filename().c_str());
            continue;
        }

        // score against tokens_new
        const llama_tokens & new_toks = tokens_new.get_tokens();
        int lcp = 0;
        const int n_min = std::min((int) toks.size(), (int) new_toks.size());
        while (lcp < n_min && toks[lcp] == new_toks[lcp]) {
            lcp++;
        }

        const float f_keep_cur = float(lcp) / float(toks.size());
        const float sim_cur    = tokens_new.size() > 0 ? float(lcp) / float(tokens_new.size()) : 0.0f;

        SRV_WRN("REMOVE_ME try_match_disk: '%s' n_tok=%u main_size=%llu drft_size=%llu lcp=%d f_keep=%.3f sim=%.3f\n",
                entry.path().filename().c_str(), *n_tok,
                (unsigned long long) *main_size, (unsigned long long) *drft_size,
                lcp, f_keep_cur, sim_cur);

        if (f_keep_cur < 0.25f) { ++n_rejected_score; continue; }

        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;
            best = disk_match{
                entry.path(),
                std::move(toks),
                *main_size,
                *drft_size,
                main_offset,
                f_keep_cur,
                sim_cur,
            };
        }
    }

    SRV_WRN("REMOVE_ME try_match_disk: scan summary — scanned=%d hdr_rej=%d hash_rej=%d cap_rej=%d asym_rej=%d score_rej=%d winner=%s\n",
            n_scanned, n_rejected_header, n_rejected_hash, n_rejected_capacity, n_rejected_asym, n_rejected_score,
            best ? best->path.filename().c_str() : "(none)");

    if (!best) {
        return false;
    }
    SRV_WRN("REMOVE_ME try_match_disk: winner '%s' n_tok=%zu main_size=%llu drft_size=%llu f_keep=%.3f sim=%.3f\n",
            best->path.filename().c_str(), best->tokens.size(),
            (unsigned long long) best->main_size, (unsigned long long) best->drft_size,
            best->f_keep, best->sim);

    // Re-open the chosen file and read the state blobs.
    // Layout from best->main_offset onward: [main_bytes][drft_size: u64][drft_bytes]
    //                                       [n_ckpts: u32][per-ckpt: n_tokens,i64; pos_min,i32; pos_max,i32;
    //                                       data_tgt_size,u64; data_tgt; data_dft_size,u64; data_dft]
    std::ifstream f(best->path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(best->main_offset);

    std::vector<uint8_t> main_bytes(static_cast<size_t>(best->main_size));
    if (best->main_size > 0) {
        f.read(reinterpret_cast<char *>(main_bytes.data()),
               static_cast<std::streamsize>(best->main_size));
        if (!f) return false;
    }

    // drft_size is always present (u64); skip past it (and any bytes) regardless of whether the
    // file carries draft data. The asymmetric-file rule (file-no-drft × runtime-has-drft) was
    // enforced during the scan phase above, so we can trust the file's drft_size here.
    uint64_t drft_sz_read = 0;
    f.read(reinterpret_cast<char *>(&drft_sz_read), sizeof(drft_sz_read));
    if (!f) return false;

    std::vector<uint8_t> drft_bytes;
    if (drft_sz_read > 0) {
        drft_bytes.resize(static_cast<size_t>(drft_sz_read));
        f.read(reinterpret_cast<char *>(drft_bytes.data()),
               static_cast<std::streamsize>(drft_sz_read));
        if (!f) return false;
    }

    // checkpoints section
    uint32_t n_ckpts = 0;
    f.read(reinterpret_cast<char *>(&n_ckpts), sizeof(n_ckpts));
    if (!f) return false;

    std::list<common_prompt_checkpoint> restored_ckpts;
    for (uint32_t i = 0; i < n_ckpts; i++) {
        int64_t  n_tokens_v = 0;
        int32_t  pos_min_v  = 0;
        int32_t  pos_max_v  = 0;
        uint64_t tgt_sz     = 0;
        uint64_t dft_sz     = 0;

        f.read(reinterpret_cast<char *>(&n_tokens_v), sizeof(n_tokens_v));
        f.read(reinterpret_cast<char *>(&pos_min_v),  sizeof(pos_min_v));
        f.read(reinterpret_cast<char *>(&pos_max_v),  sizeof(pos_max_v));
        f.read(reinterpret_cast<char *>(&tgt_sz),     sizeof(tgt_sz));
        if (!f) return false;

        common_prompt_checkpoint cp;
        cp.n_tokens = n_tokens_v;
        cp.pos_min  = pos_min_v;
        cp.pos_max  = pos_max_v;
        if (tgt_sz > 0) {
            cp.data_tgt.resize(static_cast<size_t>(tgt_sz));
            f.read(reinterpret_cast<char *>(cp.data_tgt.data()),
                   static_cast<std::streamsize>(tgt_sz));
            if (!f) return false;
        }

        f.read(reinterpret_cast<char *>(&dft_sz), sizeof(dft_sz));
        if (!f) return false;
        if (dft_sz > 0) {
            cp.data_dft.resize(static_cast<size_t>(dft_sz));
            f.read(reinterpret_cast<char *>(cp.data_dft.data()),
                   static_cast<std::streamsize>(dft_sz));
            if (!f) return false;
        }

        SRV_WRN("REMOVE_ME try_match_disk: read checkpoint %u/%u n_tokens=%lld pos_min=%d pos_max=%d tgt_sz=%llu dft_sz=%llu\n",
                i + 1, n_ckpts, (long long) n_tokens_v, pos_min_v, pos_max_v,
                (unsigned long long) tgt_sz, (unsigned long long) dft_sz);

        restored_ckpts.push_back(std::move(cp));
    }
    f.close();
    SRV_WRN("REMOVE_ME try_match_disk: read %u checkpoints from file, file closed\n", n_ckpts);

    // Restore into the slot's seq via the same primitive the RAM path uses
    const size_t n_main = llama_state_seq_set_data_ext(ctx_tgt, main_bytes.data(),
                                                        main_bytes.size(), id_slot, 0);
    SRV_WRN("REMOVE_ME try_match_disk: set_data_ext(main) returned %zu/%zu, post pos_min=%d pos_max=%d\n",
            n_main, main_bytes.size(),
            (int) llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id_slot),
            (int) llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id_slot));
    if (n_main != main_bytes.size()) {
        SRV_WRN("disk cache: set_data_ext (main) returned %zu of %zu\n", n_main, main_bytes.size());
        return false;
    }

    if (runtime_has_dft && best->drft_size > 0) {
        const size_t n_drft = llama_state_seq_set_data_ext(ctx_dft, drft_bytes.data(),
                                                            drft_bytes.size(), id_slot, 0);
        SRV_WRN("REMOVE_ME try_match_disk: set_data_ext(drft) returned %zu/%zu, post pos_min=%d pos_max=%d\n",
                n_drft, drft_bytes.size(),
                (int) llama_memory_seq_pos_min(llama_get_memory(ctx_dft), id_slot),
                (int) llama_memory_seq_pos_max(llama_get_memory(ctx_dft), id_slot));
        if (n_drft != drft_bytes.size()) {
            SRV_WRN("disk cache: set_data_ext (drft) returned %zu of %zu\n", n_drft, drft_bytes.size());
            return false;
        }
    }

    // populate prompt.tokens with the file's tokens
    prompt.tokens.clear();
    prompt.tokens.insert(best->tokens);
    SRV_WRN("REMOVE_ME try_match_disk: post-restore prompt.tokens.size=%zu prompt.checkpoints.size_about_to_be=%zu\n",
            prompt.tokens.size(), restored_ckpts.size());

    // hand the checkpoints to the slot's prompt — the rewind logic at
    // server-context.cpp:2620-2641 walks this list to recover earlier-position state
    // for SWA / hybrid / recurrent models when the main blob's window is past pos_min_thold.
    prompt.checkpoints = std::move(restored_ckpts);

    // consumed — delete the file (cache is add-only; entries are used once)
    std::filesystem::remove(best->path, ec);

    return true;
}

void server_prompt_cache::shutdown_and_spill() {
    // Idempotent: if there's nothing left to do, return silently (no logs).
    if (!worker.joinable() && pending_spill.empty() && states.empty()) {
        return;
    }

    if (disk_path.empty()) {
        stop_flag.store(true);
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        return;
    }

    // Snapshot counts so the user sees what's coming before any blocking I/O happens.
    size_t queued_writes;
    size_t pending_now;
    size_t states_now;
    size_t bytes_in_flight;
    {
        std::lock_guard<std::mutex> lock(mtx);
        queued_writes = queue.size();
        pending_now   = pending_spill.size();
        states_now    = states.size();
        bytes_in_flight = 0;
        for (const auto & p : pending_spill) bytes_in_flight += p->size();
    }
    for (const auto & s : states) bytes_in_flight += s.size();

    SRV_WRN("disk cache: shutdown — draining %zu queued, %zu pending in-flight, %zu states "
            "(total %.1f MiB to persist)\n",
            queued_writes, pending_now, states_now, bytes_in_flight / (1024.0 * 1024.0));

    // Drain the async worker. This blocks until any in-flight write completes, then any further queued.
    stop_flag.store(true);
    cv.notify_all();
    if (worker.joinable()) {
        SRV_WRN("%s", "disk cache: waiting for async writer to finish (this may take seconds-to-minutes for large states)...\n");
        worker.join();
        SRV_WRN("%s", "disk cache: async writer drained\n");
    }

    // Now synchronously spill anything left in pending_spill (failed-async cases), then states.
    int n_done  = 0;
    int n_total = (int) states.size() + (int) pending_spill.size();
    if (n_total > 0) {
        SRV_WRN("disk cache: synchronously spilling %d remaining entries\n", n_total);
    }

    while (!pending_spill.empty()) {
        auto entry = pending_spill.front();
        pending_spill.pop_front();
        ++n_done;
        SRV_WRN("disk cache: shutdown spill %d/%d (from pending)\n", n_done, n_total);
        const std::string uuid     = gen_disk_cache_uuid();
        const std::string filepath = disk_path + "/" + uuid + ".bin";
        write_entry_to_file(filepath, *entry);
    }

    while (!states.empty()) {
        if (states.front().tokens.has_mtmd) {
            states.pop_front();
            ++n_done;
            SRV_WRN("disk cache: shutdown spill %d/%d (mtmd entry — destroyed, not spillable)\n", n_done, n_total);
            continue;
        }
        ++n_done;
        SRV_WRN("disk cache: shutdown spill %d/%d (from states)\n", n_done, n_total);
        const std::string uuid     = gen_disk_cache_uuid();
        const std::string filepath = disk_path + "/" + uuid + ".bin";
        write_entry_to_file(filepath, states.front());
        states.pop_front();
    }

    SRV_WRN("%s", "disk cache: shutdown complete\n");
}
