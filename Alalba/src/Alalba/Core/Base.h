#pragma once
#ifdef ALALBA_PLATFORM_LINUX
	
	#ifdef ALALBA_BUILD_DLL
		#define ALALBA_API __attribute__((visibility("default")))
	#else
		#define ALALBA_API
	#endif // ALALBA_BUILD_DLL
#else
	#error Alalba only support Linux!
#endif

#define ALALBA_APP_ASSERT(x, ...) { if(!(x)) { ALALBA_APP_ERROR("Assertion Failed: {0}", __VA_ARGS__); abort(); } }
#define ALALBA_CORE_ASSERT(x, ...) { if(!(x)) { ALALBA_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); abort(); } }


#define BIT(x) (1<<x)