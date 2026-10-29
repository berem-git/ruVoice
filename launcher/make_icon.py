"""Рисует иконку RuVoice (звуковая волна на синем фоне). Запуск: uv run --with pillow launcher/make_icon.py"""
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

# Вертикальный градиент в скруглённом квадрате
grad = Image.new("RGBA", (S, S))
top, bottom = (88, 150, 255), (40, 70, 200)
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = y / (S - 1)
    gd.line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=220, fill=255)
img.paste(grad, (0, 0), mask)

# Полоски звуковой волны
d = ImageDraw.Draw(img)
heights = [0.28, 0.52, 0.78, 0.52, 0.28]
bar_w, gap = 104, 52
total = len(heights) * bar_w + (len(heights) - 1) * gap
x = (S - total) // 2
for h in heights:
    bh = int(S * h * 0.75)
    y0 = (S - bh) // 2
    d.rounded_rectangle([x, y0, x + bar_w, y0 + bh], radius=bar_w // 2, fill=(255, 255, 255, 255))
    x += bar_w + gap

out = Path(__file__).with_name("ruvoice.ico")
img.save(out, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
img.resize((256, 256), Image.LANCZOS).save(Path(__file__).parent.parent / "ruvoice" / "web" / "icon.png")
print(out)
