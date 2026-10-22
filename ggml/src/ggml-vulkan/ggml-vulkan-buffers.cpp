#include "ggml-vulkan-common.h"

#if defined(__linux__)
#include <dirent.h>
#endif
#include <cstdio>

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    // GGML_VK_MMVQ_SLOPE_STATS t_alloc_us / n_alloc / alloc_bytes: vkCreateBuffer + vkAllocateMemory + bind + map;
    // under VRAM pressure the allocate is where the kernel evicts to GTT
    vk_slope_timer slope_t(VK_SLOPE_ALLOC);
    if (slope_t.p) {
        slope_t.p->alloc_bytes += size;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else {
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    return buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

// [fork, 2026-10-21] GPU PLACEMENT warning (ledger lever 8), the GPU sibling of llama's CPU PLACEMENT warnings.
//
// Why: twice in two days (2026-10-21) a card ran 5-10x slow with nothing in the log. Morning: -ts disabled the fit,
// the recurrent buffer of Vulkan3 overflowed and RADV put a 931 MiB BO in GTT (GATED_DELTA_NET 39-127 ms per step vs
// 0.2-0.46). Night: the context-parallel mirror moved device between fit probes, both XTX ended over budget, Vulkan1
// started with 1087 MiB in GTT and Vulkan3 had ~1 GB evicted to GTT while running (VRAM 23717 -> 22730 MiB, GTT
// 68 -> 1281 MiB; GDN 21.5 ms, CPY 23.5 ms per 3-row step). In both cases every allocation succeeded in a
// DEVICE_LOCAL memory type: the kernel decides residency, so the memory type flags alone do not show it.
//
// What: (a) an allocation that came back without DEVICE_LOCAL (only possible with GGML_VK_ALLOW_SYSMEM_FALLBACK or
// prefer-host devices) warns at once; (b) after every device allocation and every GGML_VK_MEM_WATCH_EVERY graphs
// (default 256, 0 = off) the device-local heaps' VK_EXT_memory_budget heapUsage is compared with heapBudget; usage >
// budget is the oversubscription that ends in GTT placement/eviction. Without the extension (b) compares our own
// live device-local bytes with the heap size at allocation time and the periodic check is skipped (zero cost).
// Each kind warns once per device and re-arms when usage drops back under the budget. UMA devices are skipped.
static std::string ggml_vk_mem_watch_last(const vk_mem_watch & w) {
    std::string s;
    for (auto it = w.last.rbegin(); it != w.last.rend(); ++it) {
        char buf[96];
        snprintf(buf, sizeof(buf), "%s%.1f MiB (%s)", s.empty() ? "" : ", ", it->first/1024.0/1024.0, it->second.c_str());
        s += buf;
    }
    return s.empty() ? "none" : s;
}

static void ggml_vk_mem_watch_init(vk_device & device, vk_mem_watch & w) {
    if (w.init) {
        return;
    }
    w.init = true;
    for (const auto & ext : device->physical_device.enumerateDeviceExtensionProperties()) {
        if (strcmp(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
            w.has_budget = true;
            break;
        }
    }
    const vk::PhysicalDeviceMemoryProperties mp = device->physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            w.heap_size += mp.memoryHeaps[i].size;
        }
    }
}

// call with w.mutex held
static void ggml_vk_mem_watch_check_locked(vk_device & device, vk_mem_watch & w, const char * why) {
    if (device->uma) {
        return;
    }
    ggml_vk_mem_watch_init(device, w);

    uint64_t usage  = 0;
    uint64_t budget = 0;
    const char * src;
    if (w.has_budget) {
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT bp;
        vk::PhysicalDeviceMemoryProperties2 mp2;
        mp2.pNext = &bp;
        device->physical_device.getMemoryProperties2(&mp2);
        for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
            if (mp2.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                usage  += bp.heapUsage[i];
                budget += bp.heapBudget[i];
            }
        }
        src = "VK_EXT_memory_budget heapUsage vs heapBudget";
    } else {
        usage  = (uint64_t) std::max<int64_t>(0, w.dev_local_bytes.load());
        budget = w.heap_size;
        src = "our device-local allocations vs heap size (no VK_EXT_memory_budget)";
    }

    if (usage <= budget) {
        w.warned_budget = false;
        return;
    }
    if (w.warned_budget) {
        return;
    }
    w.warned_budget = true;

    GGML_LOG_WARN("GPU PLACEMENT: ggml_vulkan: %s (%s): device-local memory usage %.0f MiB exceeds the budget %.0f MiB by %.0f MiB "
            "(%s, %s); the kernel places or evicts buffers to host memory (GTT) and the kernels that read them run 5-10x slower; "
            "last allocations: %s; check --fit-target / -ts / -c / extra per-device buffers (e.g. LLAMA_CP_LAYER mirror)\n",
            device->name.c_str(), device->properties.deviceName.data(), usage/1024.0/1024.0, budget/1024.0/1024.0,
            (usage - budget)/1024.0/1024.0, src, why, ggml_vk_mem_watch_last(w).c_str());
}

static void ggml_vk_mem_watch_alloc(vk_device & device, const vk_buffer & buf, size_t size, const char * label) {
    if (!buf || buf->size == 0) {
        return;
    }
    vk_mem_watch & w = device->mem_watch;
    std::lock_guard<std::mutex> lock(w.mutex);

    const bool dev_local = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (dev_local) {
        w.dev_local_bytes += (int64_t) size;
        buf->mem_watch_counted = true;
    }
    w.last.emplace_back(size, label);
    if (w.last.size() > 4) {
        w.last.pop_front();
    }

    if (!dev_local && !device->uma && !device->prefer_host_memory && !w.warned_type) {
        w.warned_type = true;
        GGML_LOG_WARN("GPU PLACEMENT: ggml_vulkan: %s (%s): a %.2f MiB %s buffer was allocated in host memory (memory type "
                "without DEVICE_LOCAL: sysmem fallback, the device-local heap is full); kernels reading it run 5-10x slower; "
                "last allocations: %s; check --fit-target / -ts / -c\n",
                device->name.c_str(), device->properties.deviceName.data(), size/1024.0/1024.0, label, ggml_vk_mem_watch_last(w).c_str());
    }

    ggml_vk_mem_watch_check_locked(device, w, "after an allocation");
}

void ggml_vk_mem_watch_free(vk_device_struct * device, size_t size, bool counted) {
    if (counted) {
        device->mem_watch.dev_local_bytes -= (int64_t) size;
    }
}

// [fork, 2026-10-22] Kernel-side GTT watch (sysfs), the half of the placement picture VK_EXT_memory_budget cannot see.
//
// Why: 2026-10-22 11:30-12:00 (ledger "289k CLIFF NAMED"), n_ctx 289024, -ub 2048, split pinned -ts 23,24,19: at the
// start of the first prompt the kernel evicted the KQ mask op's never-yet-touched OUTPUT buffer (1183842304 B =
// 289024 x 2048 x 2 B, inside the fit's 2182 MiB compute buffer) off the 7900 XT (PCIe x1, DRM card1, 0000:09:00.0)
// into GTT to place the first prompt's ~250 MB of lazy allocations: sysfs mem_info_vram_used 17249 -> 17000 MiB,
// mem_info_gtt_used 522 -> 1925 MiB; amdgpu_gem_info showed the BO in GTT. Under pipelining it never migrated back
// (amdgpu_cs_bo_validate: a BO left in GTT drops off the evicted list; the serialized run had it back in VRAM within
// 0.5 s), and every FA on the XT read its mask over the x1 link: the prompt ran 5-8x slow. The GPU PLACEMENT warning
// above never fired: RADV's heapUsage stayed under heapBudget (19191 MiB free at start, ~17.5 GB demand) while the
// kernel moved the pages. Why TTM evicts with ~3 GB nominally free is still unexplained (ledger: OPEN).
//
// Cross-module preconditions: RADV's budget is its own accounting, blind to TTM placement/eviction; sysfs
// mem_info_* is the only runtime view of where the kernel put things (debugfs amdgpu_gem_info names the BO but needs
// root, so it is not read here). The counters are DEVICE-WIDE: every process on the card (a second llama-server, the
// desktop on card0) moves them too. Host-visible memory this process maps into the device also counts as GTT on it:
// Vulkan_Host buffers on the first device (the XT showed 400-5000 MiB depending on config), pinned staging, D2D host
// rings (VK_EXT_external_memory_host imports, if the kernel accounts userptr BOs in GTT). Hence only DELTAS after the
// baseline are reported, never absolute values, and the scratch growth lines (ggml_vk_scratch_grow_log) are the
// attribution for host-side growth.
//
// What: the device is mapped to /sys/bus/pci/devices/<bdf> (bdf from VK_EXT_pci_bus_info, same lowercase format as
// sysfs); without mem_info_gtt_used there (not Linux, not amdgpu, no extension) the watch is silently off; UMA devices
// (the iGPU, card3, if it is a vk_device at all) are skipped since GTT is their normal home. Baseline + one WARN
// line with the kernel and RADV views side by side at the device's first graph (after model/KV/compute allocation;
// at vk_device creation the fit's probes have the log redirected to DEBUG and the model is not loaded yet, so a
// baseline there would report the model load itself as growth). Then every GGML_VK_MEM_WATCH_EVERY graphs (shared
// cadence, 0 = off) two sysfs reads; WARN when GTT grew more than GGML_VK_GTT_WATCH_MIB (default 256, 0 = off) above
// the warn level (= the last warned value, lowered whenever GTT shrinks, so it is <= the previous sample: a jump
// since the previous sample always warns, and slow creep warns once it adds up). One line when GTT falls back below
// the level of the first warning (evicted BOs migrated back). At most 32 growth warnings per device.
static bool ggml_vk_sysfs_read_u64(const std::string & path, uint64_t & v) {
#if defined(__linux__)
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    unsigned long long x = 0;
    const bool ok = fscanf(f, "%llu", &x) == 1;
    fclose(f);
    if (ok) {
        v = (uint64_t) x;
    }
    return ok;
#else
    GGML_UNUSED(path);
    GGML_UNUSED(v);
    return false;
#endif
}

// path mapping helper: "<bdf>" -> "/sys/bus/pci/devices/<bdf>" if it has mem_info_gtt_used, card = first drm/card*
static bool ggml_vk_gtt_sysfs_dir(const std::string & bdf, std::string & dir, std::string & card) {
#if defined(__linux__)
    if (bdf.empty()) {
        return false;
    }
    dir = "/sys/bus/pci/devices/" + bdf;
    uint64_t v;
    if (!ggml_vk_sysfs_read_u64(dir + "/mem_info_gtt_used", v) || !ggml_vk_sysfs_read_u64(dir + "/mem_info_vram_used", v)) {
        return false;
    }
    card = "card?";
    if (DIR * d = opendir((dir + "/drm").c_str())) {
        while (dirent * e = readdir(d)) {
            if (strncmp(e->d_name, "card", 4) == 0) {
                card = e->d_name;
                break;
            }
        }
        closedir(d);
    }
    return true;
#else
    GGML_UNUSED(bdf);
    GGML_UNUSED(dir);
    GGML_UNUSED(card);
    return false;
#endif
}

static std::string ggml_vk_physical_device_bdf(vk::PhysicalDevice & pd) {
    bool ext = false;
    for (const auto & p : pd.enumerateDeviceExtensionProperties()) {
        if (strcmp(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME, p.extensionName) == 0) {
            ext = true;
            break;
        }
    }
    if (!ext) {
        return "";
    }
    vk::PhysicalDeviceProperties2 props;
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci;
    props.pNext = &pci;
    pd.getProperties2(&props);
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, (unsigned) (pci.pciFunction & 0xf));
    return buf;
}

// call with w.mutex held
static void ggml_vk_gtt_watch_init_locked(vk_device & device, vk_gtt_watch & w) {
    if (w.init) {
        return;
    }
    w.init = true;
    if (device->uma) {
        return;
    }
    w.bdf     = ggml_vk_physical_device_bdf(device->physical_device);
    w.enabled = ggml_vk_gtt_sysfs_dir(w.bdf, w.dir, w.card);
}

bool ggml_vk_gtt_watch_read(vk_device & device, uint64_t & gtt_used, uint64_t & vram_used) {
    vk_gtt_watch & w = device->gtt_watch;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        ggml_vk_gtt_watch_init_locked(device, w);
    }
    if (!w.enabled) {
        return false;
    }
    return ggml_vk_sysfs_read_u64(w.dir + "/mem_info_gtt_used", gtt_used) &&
           ggml_vk_sysfs_read_u64(w.dir + "/mem_info_vram_used", vram_used);
}

static void ggml_vk_gtt_watch_sample(vk_device & device, uint64_t n_graph) {
    static const uint64_t thr = [] {
        const char * e = getenv("GGML_VK_GTT_WATCH_MIB");
        return (uint64_t) (e ? std::max(0, atoi(e)) : 256) * 1024 * 1024;
    }();
    constexpr double MiB = 1024.0 * 1024.0;
    vk_gtt_watch & w = device->gtt_watch;
    uint64_t gtt = 0, vram = 0;
    if (!ggml_vk_gtt_watch_read(device, gtt, vram)) {
        return;
    }
    std::lock_guard<std::mutex> lock(w.mutex);

    if (!w.sampled) {
        w.sampled  = true;
        w.gtt_base = w.gtt_prev = w.gtt_first = gtt;
        w.vram_prev = w.vram_first = vram;
        // RADV's view next to the kernel's (the same numbers the fit read via ggml_backend_vk_get_device_memory)
        size_t free = 0, total = 0;
        ggml_backend_vk_get_device_memory((int) device->idx, &free, &total);
        uint64_t budget = 0, usage = 0, vram_total = 0;
        bool has_budget;
        {
            std::lock_guard<std::mutex> lock_mw(device->mem_watch.mutex);
            ggml_vk_mem_watch_init(device, device->mem_watch);
            has_budget = device->mem_watch.has_budget;
        }
        if (has_budget) {
            vk::PhysicalDeviceMemoryBudgetPropertiesEXT bp;
            vk::PhysicalDeviceMemoryProperties2 mp2;
            mp2.pNext = &bp;
            device->physical_device.getMemoryProperties2(&mp2);
            for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
                if (mp2.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    usage  += bp.heapUsage[i];
                    budget += bp.heapBudget[i];
                }
            }
        }
        ggml_vk_sysfs_read_u64(w.dir + "/mem_info_vram_total", vram_total);
        // WARN, not INFO: library INFO is demoted to TRACE and hidden at the default -lv 3 (fork log rule)
        GGML_LOG_WARN("GTT WATCH: ggml_vulkan: %s (%s) %s %s: kernel VRAM %.0f / %.0f MiB used, GTT %.0f MiB used (device-wide, "
                "all processes, incl. host-visible buffers mapped to it); RADV heapUsage %.0f MiB, heapBudget %.0f MiB, "
                "free %.0f of %.0f MiB; baseline at graph %llu, growth > %.0f MiB warns\n",
                device->name.c_str(), device->properties.deviceName.data(), w.bdf.c_str(), w.card.c_str(),
                vram/MiB, vram_total/MiB, gtt/MiB, usage/MiB, budget/MiB, free/MiB, total/MiB,
                (unsigned long long) n_graph, thr/MiB);
        return;
    }

    if (thr > 0 && gtt > w.gtt_base + thr) {
        if (w.n_warn < 32) {
            GGML_LOG_WARN("GTT WATCH: ggml_vulkan: %s (%s) %s %s: GTT grew %.0f -> %.0f MiB (+%.0f since the previous sample, "
                    "+%.0f since baseline), VRAM %.0f -> %.0f MiB, graph %llu; the kernel placed or evicted buffers to host memory "
                    "(GTT) and the kernels that read them run 5-10x slower (RADV's budget does not see this); "
                    "check --fit-target / -c / extra per-device buffers%s\n",
                    device->name.c_str(), device->properties.deviceName.data(), w.bdf.c_str(), w.card.c_str(),
                    w.gtt_prev/MiB, gtt/MiB, ((double) gtt - (double) w.gtt_prev)/MiB, ((double) gtt - (double) w.gtt_first)/MiB,
                    w.vram_prev/MiB, vram/MiB, (unsigned long long) n_graph,
                    w.n_warn == 31 ? " (last GTT growth warning for this device)" : "");
        }
        w.n_warn++;
        if (!w.armed_recovery) {
            w.armed_recovery = true;
            w.gtt_first = std::min(w.gtt_first, w.gtt_prev);   // recovery level = GTT before the first growth
        }
        w.gtt_base = gtt;
    } else if (gtt < w.gtt_base) {
        w.gtt_base = gtt;
    }
    if (w.armed_recovery && gtt <= w.gtt_first + thr / 4) {
        w.armed_recovery = false;
        GGML_LOG_WARN("GTT WATCH: ggml_vulkan: %s (%s) %s %s: GTT back down to %.0f MiB (baseline %.0f), VRAM %.0f MiB, graph %llu: "
                "the evicted buffers migrated back\n",
                device->name.c_str(), device->properties.deviceName.data(), w.bdf.c_str(), w.card.c_str(),
                gtt/MiB, w.gtt_first/MiB, vram/MiB, (unsigned long long) n_graph);
    }
    w.gtt_prev  = gtt;
    w.vram_prev = vram;
}

// [fork, 2026-10-22] Lazy scratch growth log (prealloc_* in ggml_vk_preallocate_buffers, D2D host ring regrowth).
//
// Why: the 289k first-prompt eviction above was triggered by ~250 MB of first-prompt allocations that nothing in the
// log named (VK_LOG_MEMORY is a debug-build macro). These buffers are sized lazily by the first graph that needs them,
// after the fit and after the model/KV/compute buffers, so they are exactly what lands on a card the fit left with
// little headroom. free is ggml_backend_vk_get_device_memory (RADV budget - usage, the fit's view) right after the
// growth; kernel VRAM/GTT is sysfs (device-wide). D2D rings are host memory imported into both devices: they cost
// no VRAM, they may show as GTT. Rate limit per buffer: the first 8 growths, then every 64th.
void ggml_vk_scratch_grow_log(vk_device & device, const char * which, size_t old_size, size_t new_size, uint32_t * counter) {
    if (new_size <= old_size) {
        return;
    }
    // size floor: the first production run (2026-10-22 12:20, 270k) printed ~60 lines of prealloc_y / split_k growing
    // 0.01 -> 0.04 -> 0.9 MiB per card, none of which can evict anything; what matters is MiB-scale (rings 16 -> 64
    // -> 256, prealloc_x hundreds of MiB when dequant K/V is on). GGML_VK_SCRATCH_GROW_MIB (default 4) is the floor on
    // the new size; growths below it are not counted either, so the per-buffer rate limit stays about real growth
    static const size_t floor_bytes = [] {
        const char * e = getenv("GGML_VK_SCRATCH_GROW_MIB");
        return (size_t) (e ? atof(e) : 4.0) * 1024 * 1024;
    }();
    if (new_size < floor_bytes) {
        return;
    }
    const uint32_t n = (*counter)++;
    if (n >= 8 && n % 64 != 0) {
        return;
    }
    constexpr double MiB = 1024.0 * 1024.0;
    size_t free = 0, total = 0;
    ggml_backend_vk_get_device_memory((int) device->idx, &free, &total);
    char kernel[128] = "";
    uint64_t gtt = 0, vram = 0;
    if (ggml_vk_gtt_watch_read(device, gtt, vram)) {
        snprintf(kernel, sizeof(kernel), ", kernel VRAM %.0f MiB / GTT %.0f MiB used", vram/MiB, gtt/MiB);
    }
    GGML_LOG_WARN("SCRATCH GROW: ggml_vulkan: %s (%s): %s %.2f -> %.2f MiB (growth #%u), free VRAM %.0f of %.0f MiB%s\n",
            device->name.c_str(), device->properties.deviceName.data(), which, old_size/MiB, new_size/MiB, n + 1,
            free/MiB, total/MiB, kernel);
}

void ggml_vk_mem_watch_graph(vk_device & device) {
    static const uint64_t every = [] {
        const char * e = getenv("GGML_VK_MEM_WATCH_EVERY");
        return e ? (uint64_t) std::max(0, atoi(e)) : (uint64_t) 256;
    }();
    vk_mem_watch & w = device->mem_watch;
    if (every == 0) {
        return;
    }
    const uint64_t n_graph = w.n_graphs.fetch_add(1, std::memory_order_relaxed);
    if (n_graph % every != 0) {
        return;
    }
    // one pointer test when the device has no sysfs node (after the first call), two small sysfs reads otherwise
    if (!device->gtt_watch.init || device->gtt_watch.enabled) {
        ggml_vk_gtt_watch_sample(device, n_graph);
    }
    if (w.init && !w.has_budget) {
        return;
    }
    std::lock_guard<std::mutex> lock(w.mutex);
    ggml_vk_mem_watch_check_locked(device, w, "periodic check while running");
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size, const char * label) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    ggml_vk_mem_watch_alloc(device, buf, size, label);

    return buf;
}

void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

void * ggml_vk_host_malloc(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");
    vk_buffer buf = ggml_vk_create_buffer(device, size,
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});

    if(!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        device->device.freeMemory(buf->device_memory);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    std::lock_guard<std::shared_mutex> guard(device->pinned_memory_mutex);
    device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));

    return buf->ptr;
}

void ggml_vk_host_free(vk_device& device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");
    std::lock_guard<std::shared_mutex> guard(device->pinned_memory_mutex);

    vk_buffer buf;
    size_t index;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(buf);

    device->pinned_memory.erase(device->pinned_memory.begin() + index);
}

void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    std::shared_lock<std::shared_mutex> guard(device->pinned_memory_mutex);
    buf = nullptr;
    buf_offset = 0;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}

void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    vk_slope_timer slope_t(VK_SLOPE_CPY_HOST);   // GGML_VK_MMVQ_SLOPE_STATS t_cpy_host_us
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    vk_slope_timer slope_t(VK_SLOPE_CPY_HOST);   // GGML_VK_MMVQ_SLOPE_STATS t_cpy_host_us
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    vk_slope_timer slope_t(VK_SLOPE_CPY_HOST);   // GGML_VK_MMVQ_SLOPE_STATS t_cpy_host_us (its wait is in t_wait_us)
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        {
            vk_slope_timer slope_w(VK_SLOPE_WAIT);   // GGML_VK_MMVQ_SLOPE_STATS t_wait_us
            VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        }
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    vk_slope_timer slope_t(VK_SLOPE_CPY_HOST);   // GGML_VK_MMVQ_SLOPE_STATS t_cpy_host_us
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");
    vk_slope_timer slope_t(VK_SLOPE_CPY_HOST);   // GGML_VK_MMVQ_SLOPE_STATS t_cpy_host_us (its wait is in t_wait_us)

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    const bool slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        {
            vk_slope_timer slope_w(VK_SLOPE_WAIT);   // GGML_VK_MMVQ_SLOPE_STATS t_wait_us
            VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX),
                     "vk_buffer_read_2d uma waitForFences", src->device);
        }
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        if (width == spitch && width == dpitch) {
            memcpy(dst, (const uint8_t *) src->ptr + offset, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        {
            vk_slope_timer slope_w(VK_SLOPE_WAIT);   // GGML_VK_MMVQ_SLOPE_STATS t_wait_us
            VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        }
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        {
            vk_slope_timer slope_w(VK_SLOPE_WAIT);   // GGML_VK_MMVQ_SLOPE_STATS t_wait_us
            VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        }
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    {
        vk_slope_timer slope_w(VK_SLOPE_WAIT);   // GGML_VK_MMVQ_SLOPE_STATS t_wait_us
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences", dst->device);
    }
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    const vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    try {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    return buf;
}

