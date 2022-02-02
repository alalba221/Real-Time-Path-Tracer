#include "alalbapch.h"
#include "OptixLayer.h"
#include "Alalba/Core/Application.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
// From imgui example 
namespace Alalba {
  OptixLayer::OptixLayer()
  {

  }

  OptixLayer::OptixLayer(const std::string& name)
  {

  }

  OptixLayer::~OptixLayer()
  {

  }


  void OptixLayer::OnAttach()
  {
    //cudaFree(0);
    //int numDevices;
    //cudaGetDeviceCount(&numDevices);
    ///*if (numDevices == 0)
    //  throw std::runtime_error("#osc: no CUDA capable devices found!");
    //std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;*/
    //ALALBA_CORE_ASSERT(numDevices != 0, "NO CUDA devices found");
    //OptixResult res=optixInit();
    //ALALBA_CORE_ASSERT(res == OPTIX_SUCCESS, "Optix Init failed");

  }
  void OptixLayer::OnDetach()
  {

  }
  void OptixLayer::OnImGuiRender()
  {

  }


}
