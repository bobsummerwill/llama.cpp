// The RPC client caches the alloc sizes reported by the server. Some backends need more memory
// for an op when its srcs are larger (e.g. the Metal TOP_K scratch is sized from nelements(src0)),
// so a size cached for small srcs must not be reused for larger ones.
// ref: https://github.com/ggml-org/llama.cpp/issues/30794
//
// The server runs in-process on a fake device that wraps the CPU device and reports a TOP_K
// alloc size that grows with src0, like Metal does.

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static ggml_backend_dev_t                cpu_dev;
static struct ggml_backend_device        fake_dev;
static struct ggml_backend_buffer_type   fake_buft;
static std::atomic<int>                  n_server_queries{0};

static size_t fake_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    if (tensor->op == GGML_OP_TOP_K && tensor->src[0] != nullptr) {
        n_server_queries++;
        return std::max(ggml_nbytes(tensor), 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]));
    }
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_t fake_dev_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return &fake_buft;
}

static ggml_backend_t fake_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = cpu_dev->iface.init_backend(cpu_dev, params);
    backend->device = dev; // so that the server uses fake_buft as the default buffer type
    return backend;
}

static bool wait_for_server(int port) {
    for (int i = 0; i < 600; i++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        const bool ok = connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0;
        close(fd);
        if (ok) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

// size the client reports for TOP_K(src0 = [n_cols, n_rows], k)
static size_t top_k_alloc_size(ggml_backend_buffer_type_t buft, ggml_context * ctx, int64_t n_cols, int64_t n_rows, int k) {
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows);
    ggml_tensor * top_k = ggml_top_k(ctx, src, k);
    return ggml_backend_buft_get_alloc_size(buft, top_k);
}

int main() {
    ggml_backend_load_all();

    cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    fake_dev = *cpu_dev;
    fake_dev.iface.get_buffer_type = fake_dev_get_buffer_type;
    fake_dev.iface.init_backend    = fake_dev_init_backend;

    fake_buft = *ggml_backend_dev_buffer_type(cpu_dev);
    fake_buft.iface.get_alloc_size = fake_buft_get_alloc_size;
    fake_buft.device = &fake_dev;

    const int port = 40000 + getpid() % 10000;
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);

    ggml_backend_dev_t devices[] = { &fake_dev };
    std::thread server([&]() {
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, 1, devices);
    });
    server.detach();
    GGML_ASSERT(wait_for_server(port));

    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint.c_str(), 0);
    GGML_ASSERT(buft != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 16*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    // the TOP_K output is [64, 8] in all cases, only src0 changes (e.g. a growing KV cache)
    const int64_t n_rows = 8;
    const int     k      = 64;

    // small src first, so that a cache keyed without the src shapes holds the smaller size
    const size_t small = top_k_alloc_size(buft, ctx, 256, n_rows, k);
    GGML_ASSERT(small >= 2*sizeof(int32_t)*256*n_rows);

    const size_t large = top_k_alloc_size(buft, ctx, 4096, n_rows, k);
    if (large < 2*sizeof(int32_t)*4096*n_rows) {
        fprintf(stderr, "FAIL: alloc size for the larger src0 is %zu, expected at least %zu (stale cached size?)\n",
                large, 2*sizeof(int32_t)*4096*n_rows);
        return 1;
    }

    // a smaller src must still be served from the cache, without querying the server again
    const int n_queries = n_server_queries.load();
    const size_t medium = top_k_alloc_size(buft, ctx, 1024, n_rows, k);
    GGML_ASSERT(medium >= 2*sizeof(int32_t)*1024*n_rows);
    if (n_server_queries.load() != n_queries) {
        fprintf(stderr, "FAIL: alloc size for a smaller src0 was not served from the cache\n");
        return 1;
    }

    ggml_free(ctx);
    printf("OK\n");
    return 0;
}
