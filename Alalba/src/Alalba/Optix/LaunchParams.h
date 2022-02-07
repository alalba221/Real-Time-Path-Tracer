#pragma once

typedef unsigned __int32 uint32_t;
struct LaunchParams
{
  
  uint32_t* colorBuffer;
  //vec2i     fbSize;
  unsigned int  image_width;
  unsigned int  image_height;
  int       frameID{ 0 };
  //float3   cam_eye;
  //float3   cam_u, cam_v, cam_w;
  //OptixTraversableHandle handle;
};