#pragma once
#ifdef ALALBA_PLATFORM_LINUX
extern Alalba::Application* Alalba::CreateApplication();
int main(int argc, char** argv){

	Alalba::Log::Init();
	ALALBA_CORE_WARN("Initialized Log");
	auto app = Alalba::CreateApplication();
	app->Run();
	delete app;
}

#endif