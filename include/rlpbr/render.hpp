#pragma once

#include <rlpbr/fwd.hpp>
#include <rlpbr/environment.hpp>
#include <memory>

namespace RLpbr {

struct BatchDeleter {
    void *state;
    void (*deletePtr)(void *, BatchBackend *);
    void operator()(BatchBackend *) const;
};

class RenderBatch {
public:
    using Handle = std::unique_ptr<BatchBackend, BatchDeleter>;

    RenderBatch(Handle &&backend,  std::vector<Environment> &&envs);

    inline Environment &getEnvironment(uint32_t idx) { return envs_[idx]; }
    inline Environment *getEnvironments() { return envs_.data(); }

private:
    Handle backend_;
    std::vector<Environment> envs_;
};

}
