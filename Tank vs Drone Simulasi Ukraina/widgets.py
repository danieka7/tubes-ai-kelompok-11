import pygame

from display_config import COLORS


# =============================================================================
# BAGIAN 5 -- WIDGET UI SEDERHANA (tombol, dropdown, checkbox)
# =============================================================================
class Button:
    def __init__(self, rect, label, on_click, primary=False, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.primary = primary
        self.enabled = enabled

    def draw(self, screen, font):
        if not self.enabled:
            bg = (50, 68, 42)
            fg = COLORS["text_dim"]
        elif self.primary:
            bg = COLORS["accent"]
            fg = (24, 34, 20)
        else:
            bg = (50, 68, 42)
            fg = COLORS["text_cream"]
        pygame.draw.rect(screen, bg, self.rect, border_radius=7)
        pygame.draw.rect(screen, COLORS["panel_border"], self.rect, width=1, border_radius=7)
        txt = font.render(self.label, True, fg)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_click(self, pos):
        if self.enabled and self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
