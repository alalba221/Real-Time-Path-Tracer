#pragma once
#ifdef ALALBA_PLATFORM_WINDOWS
	#if ALALBA_DYNAMIC_LINK	
		#ifdef ALALBA_BUILD_DLL
			#define ALALBA_API __declspec(dllexport)
		#else
			#define ALALBA_API __declspec(dllimport)
		#endif // ALALBA_BUILD_DLL

	#else
		#define ALALBA_API
	#endif

#else
#error ALALBA only support Windows!
#endif

#ifdef ALALBA_ENABLE_ASSERTS
#define ALALBA_APP_ASSERT(x,...){if(!x){ALALBA_APP_ERROR("Assertion Failed :{0}",__VA_ARGS__);__debugbreak();}} 
#define ALALBA_CORE_ASSERT(x,...){if(!x){ALALBA_CORE_ERROR("Assertion Failed :{0}",__VA_ARGS__);__debugbreak();}} 
#else
#define ALALBA_APP_ASSERT(x,...)
#define ALALBA_CORE_ASSERT(x,...)
#endif // ALALBA_ENABLE_ASSERTS


#define BIT(x) (1<<x)
#define ALALBA_BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)