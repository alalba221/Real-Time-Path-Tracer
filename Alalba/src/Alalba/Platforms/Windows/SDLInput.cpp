#include "alalbapch.h"
#include "SDLInput.h"

#include "Alalba/Core/Application.h"
#include <SDL2/SDL.h>

namespace Alalba {

	bool SDLInput::IsKeyPressedImpl(int keycode)
	{
		const unsigned char *state = SDL_GetKeyboardState(NULL); 
  	return (state[keycode]==1);
	}

	bool SDLInput::IsMouseButtonPressedImpl(int button)
	{
    int x, y;
    std::uint32_t buttons;
    SDL_PumpEvents();  // make sure we have the latest mouse state.
    buttons = SDL_GetMouseState(&x, &y);
		std::uint32_t MASK = 1<<(button-1);
		return ((buttons & MASK) !=0);
    //return ( (buttons & (SDL_BUTTON_LMASK || SDL_BUTTON_RMASK || SDL_BUTTON_MMASK)) != 0  );
	}

	std::pair<float, float> SDLInput::GetMousePositionImpl()
	{
	  int x, y;
    std::uint32_t buttons;
    SDL_PumpEvents();  // make sure we have the latest mouse state.
    buttons = SDL_GetMouseState(&x, &y);
    return {(float) x, (float) y};
	}

	float SDLInput::GetMouseXImpl()
	{
		auto[x, y] = GetMousePositionImpl();
		return x;
	}

	float SDLInput::GetMouseYImpl()
	{
		auto[x, y] = GetMousePositionImpl();
		return y;
	}

}