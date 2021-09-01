#pragma once
#include <memory>
#include "Base.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
namespace Alalba{
	class ALALBA_API Log
	{
	public:
		static void Init();

		inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; };
		inline static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; };
	private:

		static std::shared_ptr<spdlog::logger> s_CoreLogger;
		static std::shared_ptr<spdlog::logger> s_ClientLogger;
	};
}

#define ALALBA_CORE_TRACE(...) ::Alalba::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define ALALBA_CORE_INFO(...) ::Alalba::Log::GetCoreLogger()->info(__VA_ARGS__)
#define ALALBA_CORE_WARN(...) ::Alalba::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define ALALBA_CORE_ERROR(...) ::Alalba::Log::GetCoreLogger()->error(__VA_ARGS__)
#define ALALBA_CORE_FATAL(...) ::Alalba::Log::GetCoreLogger()->fatal(__VA_ARGS__)


#define ALALBA_APP_TRACE(...) ::Alalba::Log::GetClientLogger()->trace(__VA_ARGS__)
#define ALALBA_APP_INFO(...) ::Alalba::Log::GetClientLogger()->info(__VA_ARGS__)
#define ALALBA_APP_WARN(...) ::Alalba::Log::GetClientLogger()->warn(__VA_ARGS__)
#define ALALBA_APP_ERROR(...) ::Alalba::Log::GetClientLogger()->error(__VA_ARGS__)
#define ALALBA_APP_FATAL(...) ::Alalba::Log::GetClientLogger()->fatal(__VA_ARGS__)