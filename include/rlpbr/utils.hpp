#pragma once

#include <memory>

namespace RLpbr {

template <typename T>
struct HandleDeleter {
    constexpr HandleDeleter() noexcept = default;
    void operator()(std::remove_extent_t<T> *ptr) const;
};

template <typename T>
using Handle = std::unique_ptr<T, HandleDeleter<T>>;

template <typename T>
class DynArray {
public:
    explicit DynArray(size_t n) : ptr_(std::allocator<T>().allocate(n)), n_(n) {}

    DynArray(const DynArray &) = delete;

    DynArray(DynArray &&o)
        : ptr_(o.ptr_),
          n_(o.n_)
    {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }

    ~DynArray()
    {
        if (ptr_ == nullptr) return;

        for (size_t i = 0; i < n_; i++) {
            ptr_[i].~T();
        }
        std::allocator<T>().deallocate(ptr_, n_);
    }

    T &operator[](size_t idx) { return ptr_[idx]; }
    const T &operator[](size_t idx) const { return ptr_[idx]; }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }

    T *begin() { return ptr_; }
    T *end() { return ptr_ + n_; }
    const T *begin() const { return ptr_; }
    const T *end() const { return ptr_ + n_; }

    T &front() { return *begin(); }
    const T &front() const { return *begin(); }

    T &back() { return *(begin() + n_ - 1); }
    const T &back() const { return *(begin() + n_ - 1); }

    size_t size() const { return n_; }

private:
    T *ptr_;
    size_t n_;
};

}
