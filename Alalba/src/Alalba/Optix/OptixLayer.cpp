#include "alalbapch.h"
#include "OptixLayer.h"
#include "Alalba/Core/Application.h"
#include <optix_function_table_definition.h>
#include "stb_image_write.h"

// From imgui example 
namespace Alalba {

  static std::string readPTX(const std::string& filename)
  {
    std::ifstream inputPtx(filename);

    if (!inputPtx)
    {
      std::cerr << "ERROR: readPTX() Failed to open file " << filename << '\n';
      return std::string();
    }

    std::stringstream ptx;

    ptx << inputPtx.rdbuf();

    if (inputPtx.fail())
    {
      std::cerr << "ERROR: readPTX() Failed to read file " << filename << '\n';
      return std::string();
    }

    return ptx.str();
  }

  /*! SBT record for a raygen program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
  };

  /*! SBT record for a miss program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
  {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
  };

  OptixLayer::OptixLayer()
  {

  }

  OptixLayer::OptixLayer(const std::string& name)
  {

  }

  OptixLayer::~OptixLayer()
  {

  }

  void OptixLayer::initOptix()
  {
    
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    /*if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;*/
    ALALBA_CORE_ASSERT(numDevices != 0, "NO CUDA devices found");
    ALALBA_CORE_INFO("Found {0} CUDA devices", numDevices);
    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OptixResult res = optixInit();
    ALALBA_CORE_ASSERT(res == OPTIX_SUCCESS, "Optix Init failed");
    ALALBA_CORE_INFO("successfully initialized optix... yay!");
  }

  static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
  {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
  }

  void OptixLayer::createContext()
  {
    /// for this sample, do everything on one device
    const int deviceID = 0;
    // set device we want to run o
    ALALBA_CORE_ASSERT(cudaSetDevice(deviceID)== cudaSuccess, "cuda set device failed");
    // create a stream (for later)
    ALALBA_CORE_ASSERT(cudaStreamCreate(&stream) == cudaSuccess,"cuda create stream failed");

    cudaGetDeviceProperties(&deviceProps, deviceID);
    ALALBA_CORE_INFO("running on device:", deviceProps.name);
    
    // get current CUDA device context
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    ALALBA_CORE_ASSERT(cuRes == cudaSuccess, "Error querying current context: error code", cuRes);
    // create optix context
    ALALBA_CORE_ASSERT(optixDeviceContextCreate(cudaContext, 0, &optixContext)== OPTIX_SUCCESS," create optix context failed");

    ALALBA_CORE_ASSERT(optixDeviceContextSetLogCallback
    (optixContext, context_log_cb, nullptr, 4)== OPTIX_SUCCESS," ????");
  }

  /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
  void OptixLayer::createModule()
  {
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    std::string ptxCode = readPTX("assets/devicePrograms.ptx");;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    ALALBA_CORE_ASSERT(optixModuleCreateFromPTX(optixContext,
      &moduleCompileOptions,
      &pipelineCompileOptions,
      ptxCode.c_str(),
      ptxCode.size(),
      log, &sizeof_log,
      &module
    )== OPTIX_SUCCESS,log);
  }

  void OptixLayer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    ALALBA_CORE_ASSERT(optixProgramGroupCreate(optixContext,
      &pgDesc,
      1,
      &pgOptions,
      log, &sizeof_log,
      &raygenPGs[0]
    ) == OPTIX_SUCCESS, log);
  }

  void OptixLayer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    ALALBA_CORE_ASSERT(optixProgramGroupCreate(optixContext,
      &pgDesc,
      1,
      &pgOptions,
      log, &sizeof_log,
      &missPGs[0]
    ) == OPTIX_SUCCESS, log);
  }

  /*! does all setup for the hitgroup program(s) we are going to use */
  void OptixLayer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    ALALBA_CORE_ASSERT(optixProgramGroupCreate(optixContext,
      &pgDesc,
      1,
      &pgOptions,
      log, &sizeof_log,
      &hitgroupPGs[0]
    ) == OPTIX_SUCCESS, log);
  }

  /*! assembles the full pipeline of all programs */
  void OptixLayer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    ALALBA_CORE_ASSERT(optixPipelineCreate(optixContext,
      &pipelineCompileOptions,
      &pipelineLinkOptions,
      programGroups.data(),
      (int)programGroups.size(),
      log, &sizeof_log,
      &pipeline
    ) == OPTIX_SUCCESS, log);

    ALALBA_CORE_ASSERT(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
      pipeline,
      /* [in] The direct stack size requirement for direct
         callables invoked from IS or AH. */
      2 * 1024,
      /* [in] The direct stack size requirement for direct
         callables invoked from RG, MS, or CH.  */
      2 * 1024,
      /* [in] The continuation stack requirement. */
      2 * 1024,
      /* [in] The maximum depth of a traversable graph
         passed to trace. */
      1) == OPTIX_SUCCESS, log);
  }


  /*! constructs the shader binding table */
  void OptixLayer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
      RaygenRecord rec;
      ALALBA_CORE_ASSERT(optixSbtRecordPackHeader(raygenPGs[i], &rec)== OPTIX_SUCCESS,"SbtRecordPackHeader failed");
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }

    ///raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    //1 Allocate the miss record on the device 
    CUdeviceptr raygen_record;
    size_t raygen_record_size = sizeof(RaygenRecord) * raygenPGs.size();
    cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size);
    //2 Now copy our host record to the device
      cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        raygenPGs.data(),
        raygen_record_size,
        cudaMemcpyHostToDevice);
    sbt.raygenRecord = raygen_record;



    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
      MissRecord rec;
      ALALBA_CORE_ASSERT(optixSbtRecordPackHeader(missPGs[i], &rec) == OPTIX_SUCCESS,"");
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    ///missRecordsBuffer.alloc_and_upload(missRecords);
    //1 Allocate the miss record on the device 
    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissRecord) * missPGs.size();
    cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size);
    //2 Now copy our host record to the device
    cudaMemcpy(
      reinterpret_cast<void*>(miss_record),
      missPGs.data(),
      miss_record_size,
      cudaMemcpyHostToDevice);
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++) {
      int objectType = 0;
      HitgroupRecord rec;
      ALALBA_CORE_ASSERT(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec) == OPTIX_SUCCESS,"");
      rec.objectID = i;
      hitgroupRecords.push_back(rec);
    }
    ///hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
     //1 Allocate the miss record on the device 
    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitgroupRecord) * hitgroupPGs.size();
    cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size);
    //2 Now copy our host record to the device
    cudaMemcpy(
      reinterpret_cast<void*>(hitgroup_record),
      hitgroupPGs.data(),
      hitgroup_record_size,
      cudaMemcpyHostToDevice);
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
  }


  void OptixLayer::OnAttach()
  {
    ALALBA_CORE_TRACE("initializing optix...");
    this->initOptix();

    ALALBA_CORE_TRACE("creating optix context ...");
    this->createContext();

    ALALBA_CORE_TRACE("setting up module ...");
    this->createModule();

    ALALBA_CORE_TRACE("creating raygen programs ...");
    this->createRaygenPrograms();
    ALALBA_CORE_TRACE("creating miss programs ...");
    this->createMissPrograms();
    ALALBA_CORE_TRACE("creating hitgroup programs ...");
    this->createHitgroupPrograms();


    ALALBA_CORE_TRACE("setting up optix pipeline ...");
    this->createPipeline();

    ALALBA_CORE_TRACE("building SBT ...");
    this->buildSBT();

  }

  void OptixLayer::OnUpdate()
  {
    // resize
    launchParams.image_height = 1024;
    launchParams.image_width = 1200;
    if (colorBuffer)
    {
      cudaFree((void*)colorBuffer);
    }
    cudaMalloc(reinterpret_cast<void**>(&colorBuffer), launchParams.image_height * 
                                                       launchParams.image_width * 
                                                       sizeof(uint32_t));
    launchParams.colorBuffer = (uint32_t*)colorBuffer;

    //

    cudaMemcpy(
      reinterpret_cast<void*>(launchParamsBuffer),
      &launchParams,
      sizeof(LaunchParams),
      cudaMemcpyHostToDevice);
    launchParams.frameID++;

    ALALBA_CORE_ASSERT(optixLaunch(/*! pipeline we're launching launch: */
      pipeline, stream,
      /*! parameters and SBT */
      launchParamsBuffer,
      sizeof(LaunchParams),
      &sbt,
      /*! dimensions of the launch: */
      1200,
      1024,
      1
    ) == OPTIX_SUCCESS, "optixLaunch failed");
    cudaDeviceSynchronize();
    
    // download back to host
    std::vector<uint32_t> pixels(1200 * 1024);
    cudaMemcpy(
      pixels.data(),
      (void*)colorBuffer,
      1200 * 1024 * sizeof(uint32_t),
      cudaMemcpyDeviceToHost);
    
    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(), 1200, 1024, 4,
      pixels.data(), 1200 * sizeof(uint32_t));

  }
  void OptixLayer::OnDetach()
  {

  }
  void OptixLayer::OnImGuiRender()
  {

  }


}
