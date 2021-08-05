#pragma once

#include <rlpbr/environment.hpp>

#include <cstdint>
#include <vector>

namespace RLpbr {

struct EnvironmentBackend {};

struct LoaderBackend {};

struct BatchBackend {};

struct RenderBackend {};

template <typename LoaderType>
void destroyLoader(LoaderBackend *ptr)
{
    auto *backend_ptr = static_cast<LoaderType *>(ptr);
    delete backend_ptr;
}

template <typename LoaderType>
LoaderImpl makeLoaderImpl(LoaderBackend *ptr)
{
    return LoaderImpl(destroyLoader<LoaderType>,
        static_cast<LoaderImpl::LoadSceneType>(&LoaderType::loadScene), ptr);
}

template <typename EnvType>
void destroyEnvironment(EnvironmentBackend *ptr)
{
    auto *backend_ptr = static_cast<EnvType *>(ptr);
    delete backend_ptr;
}

template <typename EnvType>
EnvironmentImpl makeEnvironmentImpl(EnvironmentBackend *ptr)
{
    return EnvironmentImpl(destroyEnvironment<EnvType>,
        static_cast<EnvironmentImpl::AddLightType>(&EnvType::addLight),
        static_cast<EnvironmentImpl::RemoveLightType>(&EnvType::removeLight),
        ptr);
}

template <typename RendererType>
void destroyRenderer(RenderBackend *ptr)
{
    auto *backend_ptr = static_cast<RendererType *>(ptr);
    delete backend_ptr;
}

template <typename RendererType>
RendererImpl makeRendererImpl(RenderBackend *ptr)
{
    return RendererImpl(destroyRenderer<RendererType>,
        static_cast<RendererImpl::MakeLoaderType>(
            &RendererType::makeLoader),
        static_cast<RendererImpl::MakeEnvironmentType>(
            &RendererType::makeEnvironment),
        static_cast<RendererImpl::MakeBatchType>(
            &RendererType::makeRenderBatch),
        static_cast<RendererImpl::RenderType>(
            &RendererType::render),
        static_cast<RendererImpl::WaitType>(
            &RendererType::waitForBatch),
        static_cast<RendererImpl::GetOutputType>(
            &RendererType::getOutputPointer),
        static_cast<RendererImpl::GetAuxType>(
            &RendererType::getAuxiliaryOutputs),
        ptr);
}

}
