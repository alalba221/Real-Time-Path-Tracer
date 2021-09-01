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

#define BIT(x) (1<<x)