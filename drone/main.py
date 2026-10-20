import sys
import pygame

from balance_tuning import TUNE_FLAG, run_tuning_cli
from display_config import MIN_SCREEN_H, MIN_SCREEN_W, SCREEN_H, SCREEN_W
from game import Game


# =============================================================================
# BAGIAN 7 -- MAIN LOOP
# =============================================================================
def main():
    if TUNE_FLAG in sys.argv[1:]:
        run_tuning_cli(sys.argv[1:])
        return

    pygame.init()
    pygame.display.set_caption("Tank vs Drone Simulasi Ukraina")

    # RESIZABLE: jendela boleh ditarik / di-maximize. Ukuran layar
    # sebenarnya ikut berubah, bukan sekadar gambar yang diperbesar.
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    game = Game(screen)

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                # Jendela ditarik / di-maximize / fullscreen.
                # Kalau ditarik lebih kecil dari minimum, paksa kembali.
                if event.w < MIN_SCREEN_W or event.h < MIN_SCREEN_H:
                    pygame.display.set_mode(
                        (max(MIN_SCREEN_W, event.w), max(MIN_SCREEN_H, event.h)),
                        pygame.RESIZABLE,
                    )
                # Hitung ulang tata letak untuk kedua mode (eksplorasi & battle).
                w, h = pygame.display.get_surface().get_size()
                game.on_resize(w, h)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()  # memicu VIDEORESIZE juga
                else:
                    game.handle_keydown(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                game.handle_click(event.pos)

        game.update(now_ms)
        game.draw()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
