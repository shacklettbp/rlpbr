#pragma once

#include <glm/glm.hpp>
#include <memory>

namespace RLpbr {
namespace editor {

struct FreeDeleter {
    inline void operator()(void *ptr) const { return free(ptr); }
};

using UniqueMallocPtr = std::unique_ptr<void, FreeDeleter>;

}
}
