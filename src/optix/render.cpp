#include "render.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <rlpbr_core/utils.hpp>

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <iostream>

using namespace std;

namespace RLpbr {
namespace optix {

static void optixLog(unsigned int level, const char *tag,
                     const char *message, void *)
{
    cerr << "[" << setw(2) << level << "][" << setw(12)
         << tag << "]: " << message << "\n";
}

static OptixDeviceContext initializeOptix(uint32_t gpu_id, bool validate)
{
    REQ_CUDA(cudaSetDevice(gpu_id));
    REQ_CUDA(cudaFree(nullptr));

    // FIXME Drop stubs
    REQ_OPTIX(optixInit());

    OptixDeviceContextOptions optix_opts;
    if (validate) {
        optix_opts.logCallbackFunction = &optixLog;
        optix_opts.logCallbackLevel = 4;
        optix_opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    } else {
        optix_opts.logCallbackFunction = nullptr;
        optix_opts.logCallbackLevel = 0;
    }
    optix_opts.logCallbackData = nullptr;

    CUcontext cuda_ctx = nullptr;
    OptixDeviceContext optix_ctx;
    REQ_OPTIX(optixDeviceContextCreate(cuda_ctx, &optix_opts, &optix_ctx));
    REQ_OPTIX(optixDeviceContextSetCacheEnabled(optix_ctx, false));

    return optix_ctx;
}

static vector<char> compileToPTX(const char *cu_path, int gpu_id,
                                 const vector<string> &extra_options,
                                 bool validate)
{
    ifstream cu_file(cu_path, ios::binary | ios::ate);
    size_t num_cu_bytes = cu_file.tellg();
    cu_file.seekg(ios::beg);

    vector<char> cu_src(num_cu_bytes + 1);
    cu_file.read(cu_src.data(), num_cu_bytes);
    cu_file.close();
    cu_src[num_cu_bytes] = '\0';

    nvrtcProgram prog;
    REQ_NVRTC(nvrtcCreateProgram(&prog, cu_src.data(), cu_path, 0,
                                 nullptr, nullptr));

    vector<const char *> nvrtc_options = {
        NVRTC_OPTIONS
    };

    cudaDeviceProp dev_props;
    REQ_CUDA(cudaGetDeviceProperties(&dev_props, gpu_id));
    
    string arch_str = "compute_" + to_string(dev_props.major) + to_string(dev_props.minor);
    nvrtc_options.push_back("-arch");
    nvrtc_options.push_back(arch_str.c_str());

    for (const string &extra : extra_options) {
        nvrtc_options.push_back(extra.c_str());
    }

    if (validate) {
        nvrtc_options.push_back("--device-debug");
    } else {
        nvrtc_options.push_back("--extra-device-vectorization");
    }

    nvrtcResult res = nvrtcCompileProgram(prog, nvrtc_options.size(),
        nvrtc_options.data());

    auto print_compile_log = [&prog]() {
        // Retrieve log output
        size_t log_size = 0;
        REQ_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));

        if (log_size > 1) {
            vector<char> nvrtc_log(log_size);
            REQ_NVRTC(nvrtcGetProgramLog(prog, &nvrtc_log[0]));
            cerr << nvrtc_log.data() << endl;
        }

    };

    if (res != NVRTC_SUCCESS) {
        cerr << "NVRTC compilation failed" << endl;
        print_compile_log();
        abort();
    } else if (validate) {
        print_compile_log();
    }

    size_t num_ptx_bytes;
    REQ_NVRTC(nvrtcGetPTXSize(prog, &num_ptx_bytes));

    vector<char> ptx_data(num_ptx_bytes);

    REQ_NVRTC(nvrtcGetPTX(prog, ptx_data.data()));

    REQ_NVRTC(nvrtcDestroyProgram(&prog));

    return ptx_data;
}

static Pipeline buildPipeline(OptixDeviceContext ctx, const RenderConfig &cfg,
                              const ShaderBuffers &base_buffers, bool validate)
{
    OptixModuleCompileOptions module_compile_options {};
    if (validate) {
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    } else {
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    }

    OptixPipelineCompileOptions pipeline_compile_options {};
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "launchInput";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    string sampling_define;
    if (uint64_t(cfg.spp) * uint64_t(cfg.imgWidth) * uint64_t(cfg.imgHeight) >
        uint64_t(~0u)) {
        cout << "Warning, SPP too high, falling back to uniform sampling"
             << endl;
        sampling_define = "-DUNIFORM_SAMPLING";
    } else {
        sampling_define = "-DZSOBOL_SAMPLING";
    }
    
    if (cfg.flags & RenderFlags::ForceUniform) {
        sampling_define = "-DUNIFORM_SAMPLING";
    }

    vector extra_compile_options {
        string("-DSPP=(") + to_string(cfg.spp) + "u)",
        string("-DMAX_DEPTH=(") + to_string(cfg.maxDepth) + "u)",
        string("-DRES_X=(") + to_string(cfg.imgWidth) + "u)",
        string("-DRES_Y=(") + to_string(cfg.imgHeight) + "u)",
        string("-DOUTPUT_PTR=(") +
               to_string((uintptr_t)base_buffers.outputBuffer) + "ul)",
        string("-DENV_PTR=(") +
               to_string((uintptr_t)base_buffers.envs) + "ul)",
        move(sampling_define),
    };

    if (cfg.flags & RenderFlags::AuxiliaryOutputs) {
        extra_compile_options.emplace_back("-DAUXILIARY_OUTPUTS");
        extra_compile_options.emplace_back(string("-DNORMAL_PTR=(") +
                to_string((uintptr_t)base_buffers.normalBuffer) + "ul)");
        extra_compile_options.emplace_back(string("-DALBEDO_PTR=(") +
                to_string((uintptr_t)base_buffers.albedoBuffer) + "ul)");
    }
    if (cfg.clampThreshold > 0.f) {
        extra_compile_options.emplace_back(
            string("-DINDIRECT_CLAMP=(") + to_string(cfg.clampThreshold) + "f)");
    }

    vector<char> ptx = compileToPTX(STRINGIFY(OPTIX_SHADER), cfg.gpuID,
        extra_compile_options, validate);

    static constexpr size_t log_bytes = 2048;
    static char log_str[log_bytes];
    size_t log_bytes_written = log_bytes;
    
    OptixModule shader_module;
    REQ_OPTIX(optixModuleCreateFromPTX(ctx, &module_compile_options,
        &pipeline_compile_options, ptx.data(), ptx.size(),
        log_str, &log_bytes_written, &shader_module));

    OptixProgramGroup raygen_group;
    OptixProgramGroup miss_group;
    OptixProgramGroup hit_group;
    
    OptixProgramGroupOptions program_group_options {};
    OptixProgramGroupDesc raygen_prog_group_desc {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = shader_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    REQ_OPTIX(optixProgramGroupCreate(
        ctx,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &raygen_group));
    
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = shader_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    REQ_OPTIX(optixProgramGroupCreate(
        ctx,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &miss_group));
    
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = shader_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    REQ_OPTIX(optixProgramGroupCreate(
        ctx,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &hit_group));

    array program_groups {
        raygen_group,
        miss_group,
        hit_group,
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    if (validate) {
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    } else {
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    }
    
    OptixPipeline pipeline = nullptr;
    REQ_OPTIX(optixPipelineCreate(
        ctx,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(),
        nullptr,
        nullptr,
        &pipeline));

    return Pipeline {
        shader_module,
        move(program_groups),
        pipeline,
    };
}

static SBT buildSBT(cudaStream_t strm, const Pipeline &pipeline)
{
    struct EmptyEntry {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    CUdeviceptr storage = (CUdeviceptr)allocCU(sizeof(EmptyEntry) * 3);

    array<CUdeviceptr, 3> offsets;

    CUdeviceptr cur_ptr = storage;
    for (uint32_t i = 0; i < 3; i++) {
        offsets[i] = cur_ptr;

        EmptyEntry stage_entry;
        REQ_OPTIX(optixSbtRecordPackHeader(pipeline.groups[i], &stage_entry));

        REQ_CUDA(cudaMemcpyAsync((void *)cur_ptr, &stage_entry,
            sizeof(EmptyEntry), cudaMemcpyHostToDevice, strm));

        cur_ptr += sizeof(EmptyEntry);
    }

    OptixShaderBindingTable sbt;
    sbt.raygenRecord = offsets[0];
    sbt.exceptionRecord = 0;

    sbt.missRecordBase = offsets[1];
    sbt.missRecordStrideInBytes = sizeof(EmptyEntry);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = offsets[2];
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyEntry);
    sbt.hitgroupRecordCount = 1;

    sbt.callablesRecordBase = 0;
    sbt.callablesRecordStrideInBytes = 0;
    sbt.callablesRecordCount = 0;

    return SBT {
        storage,
        sbt,
    };
}

static RenderState makeRenderState(const RenderConfig &cfg,
                                   uint32_t num_frames)
{
    uint64_t env_batch_bytes = sizeof(PackedEnv) * cfg.batchSize;
    uint64_t instance_bytes =
        sizeof(PackedInstance) * Config::maxInstances;
    uint64_t transform_bytes =
        sizeof(PackedTransforms) * Config::maxInstances;
    uint64_t instance_material_bytes =
        sizeof(uint32_t) * Config::maxInstanceMaterials;
    uint64_t light_bytes =
        sizeof(PackedLight) * Config::maxLights;

    uint64_t launch_input_offset = alignOffset(
        env_batch_bytes * num_frames, 16);

    uint64_t instance_offset = alignOffset(
        launch_input_offset + sizeof(LaunchInput) * num_frames, 16);
    uint64_t transform_offset = alignOffset(
        instance_offset + instance_bytes * num_frames, 16);
    uint64_t instance_material_offset = alignOffset(
        transform_offset + transform_bytes * num_frames, 16);
    uint64_t light_offset = alignOffset(
        instance_material_offset + instance_material_bytes * num_frames, 16);

    uint64_t total_param_bytes = light_offset + light_bytes * num_frames;

    void *param_buffer;
    REQ_CUDA(cudaHostAlloc(&param_buffer, total_param_bytes,
        cudaHostAllocMapped | cudaHostAllocWriteCombined));

    bool aux_enabled = cfg.flags & RenderFlags::AuxiliaryOutputs;

    RenderState state {
        (half *)allocCU(sizeof(half) * 4 * cfg.batchSize * cfg.imgHeight *
                        cfg.imgWidth * num_frames),
        aux_enabled ? 
            (half *)allocCU(sizeof(half) * 3 * cfg.batchSize * cfg.imgHeight *
                            cfg.imgWidth * num_frames) : nullptr,
        aux_enabled ?
            (half *)allocCU(sizeof(half) * 3 * cfg.batchSize * cfg.imgHeight *
                            cfg.imgWidth * num_frames) : nullptr,
        param_buffer,
        {},
    };

    for (uint32_t frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        char *param_base = (char *)param_buffer;

        half *output_ptr = state.output + 4 * frame_idx * cfg.batchSize *
                cfg.imgHeight * cfg.imgWidth;

        PackedEnv *env_ptr = (PackedEnv *)(
            param_base + env_batch_bytes * frame_idx);

        LaunchInput *launch_input_ptr = (LaunchInput *)(param_base +
            launch_input_offset + sizeof(LaunchInput) * frame_idx);

        PackedInstance *instance_ptr = (PackedInstance *)(param_base +
            instance_offset + instance_bytes * frame_idx);

        PackedTransforms *transform_ptr = (PackedTransforms *)(param_base +
            transform_offset + transform_bytes * frame_idx);

        uint32_t *instance_material_ptr = (uint32_t *)(param_base +
            instance_material_offset + instance_material_bytes * frame_idx);

        PackedLight *light_ptr = (PackedLight *)(param_base +
            light_offset + light_bytes * frame_idx);

        half *normal_ptr = nullptr;
        half *albedo_ptr = nullptr;

        if (aux_enabled) {
            normal_ptr = state.normal + 3 * frame_idx * cfg.batchSize *
                cfg.imgHeight * cfg.imgWidth;
            albedo_ptr = state.albedo + 3 * frame_idx * cfg.batchSize *
                cfg.imgHeight * cfg.imgWidth;
        }

        state.shaderBuffers[frame_idx] = {
            output_ptr,
            normal_ptr,
            albedo_ptr,
            env_ptr,
            launch_input_ptr,
            instance_ptr,
            transform_ptr,
            instance_material_ptr,
            light_ptr,
        };
    }

    return state;
}

static cudaStream_t makeStream()
{
    cudaStream_t strm;
    REQ_CUDA(cudaStreamCreate(&strm));

    return strm;
}

static inline uint32_t getNumFrames(const RenderConfig &)
{
    return 1;
}

static BSDFLookupTables loadBSDFLookupTables(TextureManager &tex_mgr,
                                             cudaStream_t strm)
{
    const string dir = STRINGIFY(RLPBR_DATA_DIR);

    string diffuse_avg_path = dir + "/diffuse_avg_albedo.bin";
    constexpr glm::u32vec2 diffuse_avg_dims(16, 16);
    array<float, diffuse_avg_dims.x * diffuse_avg_dims.y> diffuse_avg;
    ifstream diffuse_avg_file(diffuse_avg_path);
    diffuse_avg_file.read((char *)diffuse_avg.data(),
                          diffuse_avg.size() * sizeof(float));

    Texture diffuse_avg_tex = tex_mgr.load(diffuse_avg_path,
        TextureFormat::R32_SFLOAT, cudaAddressModeClamp, strm,
        [&](const auto &) {
            return make_tuple(diffuse_avg.data(), diffuse_avg_dims, 1, 0.f);
        });

    string diffuse_dir_path = dir + "/diffuse_dir_albedo.bin";
    constexpr glm::u32vec3 diffuse_dir_dims(16, 16, 16);
    array<float, diffuse_dir_dims.x * diffuse_dir_dims.y * diffuse_dir_dims.z> 
        diffuse_dir;
    ifstream diffuse_dir_file(diffuse_dir_path);
    diffuse_dir_file.read((char *)diffuse_dir.data(),
                          diffuse_dir.size() * sizeof(float));

    Texture diffuse_dir_tex = tex_mgr.load(diffuse_dir_path,
        TextureFormat::R32_SFLOAT, cudaAddressModeClamp, strm,
        [&](const auto &) {
            return make_tuple(diffuse_dir.data(), diffuse_dir_dims, 1, 0.f);
        });

    string ggx_avg_path = dir + "/ggx_avg_albedo.bin";
    constexpr glm::u32vec1 ggx_avg_dims(32);
    array<float, ggx_avg_dims.x> ggx_avg;
    ifstream ggx_avg_file(ggx_avg_path);
    ggx_avg_file.read((char *)ggx_avg.data(),
                        ggx_avg.size() * sizeof(float));

    Texture ggx_avg_tex = tex_mgr.load(ggx_avg_path,
        TextureFormat::R32_SFLOAT, cudaAddressModeClamp, strm,
        [&](const auto &) {
            return make_tuple(ggx_avg.data(), ggx_avg_dims, 1, 0.f);
        });

    string ggx_dir_path = dir + "/ggx_dir_albedo.bin";
    constexpr glm::u32vec2 ggx_dir_dims(32, 32);
    array<float, ggx_dir_dims.x * ggx_dir_dims.y> ggx_dir;
    ifstream ggx_dir_file(ggx_dir_path);
    ggx_dir_file.read((char *)ggx_dir.data(),
                      ggx_dir.size() * sizeof(float));

    Texture ggx_dir_tex = tex_mgr.load(ggx_dir_path,
        TextureFormat::R32_SFLOAT, cudaAddressModeClamp, strm,
        [&](const auto &) {
            return make_tuple(ggx_dir.data(), ggx_dir_dims, 1, 0.f);
        });

    string ggx_inv_path = dir + "/ggx_dir_inv.bin";
    constexpr glm::u32vec2 ggx_inv_dims(128, 32);
    array<float, ggx_inv_dims.x * ggx_inv_dims.y> ggx_inv;
    ifstream ggx_inv_file(ggx_inv_path);
    ggx_inv_file.read((char *)ggx_inv.data(),
                      ggx_inv.size() * sizeof(float));

    Texture ggx_inv_tex = tex_mgr.load(ggx_inv_path,
        TextureFormat::R32_SFLOAT, cudaAddressModeClamp, strm,
        [&](const auto &) {
            return make_tuple(ggx_inv.data(), ggx_inv_dims, 1, 0.f);
        });

    BSDFPrecomputed device_hdls {
        diffuse_avg_tex.getHandle(),
        diffuse_dir_tex.getHandle(),
        ggx_avg_tex.getHandle(),
        ggx_dir_tex.getHandle(),
        ggx_inv_tex.getHandle(),
    };

    return BSDFLookupTables {
        move(diffuse_avg_tex),
        move(diffuse_dir_tex),
        move(ggx_avg_tex),
        move(ggx_dir_tex),
        move(ggx_inv_tex),
        device_hdls,
    };
}

static optional<PhysicsSimulator> makePhysicsSimulator(const RenderConfig &cfg)
{
    if (!(cfg.flags & RenderFlags::EnablePhysics)) {
        return optional<PhysicsSimulator>();
    }

    return PhysicsSimulator(PhysicsConfig {
        cfg.batchSize,
    });
}

OptixBackend::OptixBackend(const RenderConfig &cfg, bool validate)
    : ctx_(initializeOptix(cfg.gpuID, validate)), // Needs to be first
      batch_size_(cfg.batchSize),
      img_dims_(cfg.imgWidth, cfg.imgHeight),
      active_idx_(0),
      frame_counter_(0),
      frame_mask_(getNumFrames(cfg) == 2 ? 1 : 0),
      streams_([&cfg]() {
          cudaStream_t strm = makeStream();

          cudaStream_t strm2 = nullptr;
          if (getNumFrames(cfg) > 1) {
              strm2 = makeStream();
          }

          return array<cudaStream_t, 2>{strm, strm2};
      }()),
      tlas_strm_(makeStream()),
      render_state_(makeRenderState(cfg, getNumFrames(cfg))),
      pipeline_(
          buildPipeline(ctx_, cfg, render_state_.shaderBuffers[0], validate)),
      sbt_(buildSBT(streams_[0], pipeline_)),
      texture_mgr_(),
      bsdf_luts_(loadBSDFLookupTables(texture_mgr_, tlas_strm_)),
      max_texture_resolution_(cfg.maxTextureResolution == 0 ? ~0u :
                              cfg.maxTextureResolution),
      physics_(makePhysicsSimulator(cfg))
{
    REQ_CUDA(cudaStreamSynchronize(streams_[0]));

    if (cfg.imgHeight > 65535 || cfg.imgWidth > 65535) {
        cerr << "Max resolution is 65535 in either dimension" << endl;
        abort();
    }

    if (cfg.spp == 0 || (cfg.spp & (cfg.spp - 1)) != 0) {
        cerr << "Only power of 2 samples per pixel are supported" << endl;
        abort();
    }

    for (int i = 0; i < (int)getNumFrames(cfg); i++) {
        render_state_.shaderBuffers[i].launchInput->precomputed =
            bsdf_luts_.deviceHandles;
    }
}

LoaderImpl OptixBackend::makeLoader()
{
    OptixLoader *loader = new OptixLoader(ctx_, texture_mgr_,
                                          max_texture_resolution_,
                                          physics_.has_value());
    return makeLoaderImpl<OptixLoader>(loader);
}

EnvironmentImpl OptixBackend::makeEnvironment(const shared_ptr<Scene> &scene,
                                              const Camera &)
{
    const OptixScene &optix_scene = *static_cast<OptixScene *>(scene.get());
    OptixEnvironment *environment = new OptixEnvironment(
        OptixEnvironment::make(ctx_, tlas_strm_, optix_scene));
    return makeEnvironmentImpl<OptixEnvironment>(environment);
}

RenderBatch::Handle OptixBackend::makeRenderBatch()
{
    auto batch_deleter = [](void *, BatchBackend *) {};

    return RenderBatch::Handle(nullptr, {nullptr, batch_deleter});
}

static array<float4, 3> packCamera(const Camera &cam)
{
    array<float4, 3> packed;

    glm::vec3 scaled_up = -cam.tanFOV * cam.up;
    glm::vec3 scaled_right = cam.aspectRatio * cam.tanFOV * cam.right;

    packed[0] = make_float4(cam.position.x, cam.position.y,
        cam.position.z, cam.view.x);
    packed[1] = make_float4(cam.view.y, cam.view.z,
        scaled_up.x, scaled_up.y);
    packed[2] = make_float4(scaled_up.z, scaled_right.x,
        scaled_right.y, scaled_right.z);

    return packed;
}

static PackedEnv packEnv(const Environment &env,
                         PackedInstance **instance_buffer,
                         PackedTransforms **instance_transforms,
                         uint32_t **instance_materials,
                         PackedLight **light_buffer)
{
    const OptixEnvironment &env_backend =
        *static_cast<const OptixEnvironment *>(env.getBackend());
    const OptixScene &scene = 
        *static_cast<const OptixScene *>(env.getScene().get());
    
    PackedInstance *cur_instance = *instance_buffer;
    PackedInstance *env_inst_start = cur_instance;
    for (const auto &inst : env.getInstances()) {
        *cur_instance++ = PackedInstance {
            inst.materialOffset,
            scene.objectInfo[inst.objectIndex].meshIndex,
        };
    }
    *instance_buffer = cur_instance;

    PackedTransforms *env_txfm_start = *instance_transforms;
    memcpy(*instance_transforms, env.getTransforms().data(),
           sizeof(PackedTransforms) * env.getTransforms().size());
    *instance_transforms += env.getTransforms().size();

    uint32_t *cur_inst_materials = *instance_materials;
    uint32_t *env_mat_start = cur_inst_materials;
    for (uint32_t mat_idx : env.getInstanceMaterials()) {
        *cur_inst_materials++ = mat_idx;
    }
    *instance_materials = cur_inst_materials;

    (void)light_buffer;
    return PackedEnv {
        packCamera(env.getCamera()),
        env_backend.tlas.hdl,
        scene.vertexPtr,
        scene.indexPtr,
        scene.materialPtr,
        scene.texturePtr,
        scene.textureDimsPtr,
        scene.meshPtr,
        env_inst_start,
        env_mat_start,
        env_backend.lights,
        (PackedTransforms *)env_txfm_start,
        env_backend.numLights,
    };
}

void OptixBackend::render(RenderBatch &batch)
{
    const Environment *envs = batch.getEnvironments();
    //physics_->simulate(envs);

    ShaderBuffers &buffers = render_state_.shaderBuffers[active_idx_];
    PackedInstance *instance_buffer = buffers.instanceBuffer;
    PackedTransforms *transform_buffer = buffers.transformBuffer;
    uint32_t *instance_material_buffer = buffers.instanceMaterialBuffer;
    PackedLight *light_buffer = buffers.lightBuffer;

    for (int batch_idx = 0; batch_idx < (int)batch_size_; batch_idx++) {
        const Environment &env = envs[batch_idx];
        if (env.isDirty()) {
            OptixEnvironment *env_backend = (OptixEnvironment *)env.getBackend();
            env_backend->queueTLASRebuild(env, ctx_, streams_[active_idx_]);
            env.clearDirty();
        }

        buffers.envs[batch_idx] =
            packEnv(env, &instance_buffer, &transform_buffer,
                    &instance_material_buffer, &light_buffer);
    }

    buffers.launchInput->baseBatchOffset = batch_size_ * active_idx_;
    buffers.launchInput->baseFrameCounter = frame_counter_;

    REQ_OPTIX(optixLaunch(pipeline_.hdl, streams_[active_idx_],
        (CUdeviceptr)buffers.launchInput, sizeof(LaunchInput),
        &sbt_.hdl, img_dims_.x, img_dims_.y, batch_size_));

    REQ_CUDA(cudaStreamSynchronize(streams_[active_idx_]));

    frame_counter_ += batch_size_;

    active_idx_ = (active_idx_ + 1) & frame_mask_;
}

void OptixBackend::waitForBatch(RenderBatch &)
{
    REQ_CUDA(cudaStreamSynchronize(streams_[0]));
}

half *OptixBackend::getOutputPointer(RenderBatch &)
{
    return render_state_.shaderBuffers[0].outputBuffer;
}

AuxiliaryOutputs OptixBackend::getAuxiliaryOutputs(RenderBatch &)
{
    return {
        render_state_.shaderBuffers[0].normalBuffer,
        render_state_.shaderBuffers[0].albedoBuffer,
    };
}

}
}
