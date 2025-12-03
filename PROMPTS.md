# E-Motion Prompt Documentation

This document lists all the prompts currently used in the application, categorized by style and emotion.

## 1. Base Prompts
These are the starting points for each style before emotion is injected. Defined in `diffusion/config.py`.

| Style | Base Prompt |
| :--- | :--- |
| **Abstract** | `vibrant abstract expressionist painting, dynamic brushstrokes, emotional energy` |
| **Realistic** | `photorealistic portrait, natural lighting, detailed face` |
| **Combo (Sci-Fi)** | `artistic portrait, expressive lighting, emotional atmosphere, detailed face, high quality, 8k resolution` |

---

## 2. Emotion Conditioned (Abstract)
**Method:** Text Prompt + Projection Layer (if enabled)
**Logic:** `Base Prompt` + `, ` + `Rich Description` + `, abstract representation of {emotion}`

| Emotion | Rich Description (Injected) |
| :--- | :--- |
| **Angry** | `violent red and black oil painting, chaotic brushstrokes, sharp jagged lines, exploding composition, aggressive texture, storm-like fury, intense contrast, visual noise` |
| **Disgust** | `sickly green and muddy brown, distorted melting shapes, uneven oozing texture, repelling composition, murky atmosphere, unsettling asymmetry, biological horror style` |
| **Fear** | `cold dark blue and charcoal grey, trembling thin lines, claustrophobic composition, deep shadows, nervous energy, mysterious fog, sharp angles, psychological horror` |
| **Happy** | `vibrant yellow and warm orange, flowing organic curves, radiant sunbursts, harmonious golden ratio composition, joyful energy, soft blooming textures, uplifting atmosphere` |
| **Neutral** | `balanced beige and soft grey, minimalist geometric shapes, calm horizontal lines, steady rhythm, peaceful symmetry, structured composition, zen garden aesthetic` |
| **Sad** | `melancholic deep blue and rain grey, heavy downward dripping strokes, lonely negative space, quiet composition, faded watercolor texture, tear-stained atmosphere` |
| **Surprise** | `electric neon colors, dynamic radial explosion, sudden shockwaves, high contrast pop art style, vibrant energy burst, erratic patterns, visual impact` |

---

## 3. Emotion & Image Conditioned (Combo)
**Method:** ControlNet (Canny) + Emotion Atmosphere
**Logic:** `Base Prompt` + `, ` + `Atmosphere Modifier` + `, {emotion} facial expression, highly expressive {emotion} emotion`
**Negative Prompt:** `low quality, blurry, sketch, drawing, bad anatomy, distorted face`

| Emotion | Atmosphere Modifier (Injected) |
| :--- | :--- |
| **Happy** | `radiant golden light, blooming flowers, warm sunbeams, floating petals, soft glowing aura, harmonious atmosphere, joyful energy` |
| **Sad** | `falling rain, blue melancholic tones, heavy shadows, weeping willow textures, cold atmosphere, lonely composition, tear-stained glass effect` |
| **Angry** | `burning fire, exploding sparks, jagged red lightning, aggressive storm clouds, intense heat haze, chaotic energy, cracked textures` |
| **Surprise** | `sudden burst of confetti, electric shockwaves, vibrant pop-art colors, dynamic motion blur, wide-angle distortion, energetic splash` |
| **Fear** | `creeping shadows, dark fog, trembling lines, cold pale light, claustrophobic vignette, mysterious silhouettes, nervous atmosphere` |
| **Disgust** | `sickly green haze, distorted melting textures, uneven lighting, repelling composition, murky sludge, unsettling organic shapes` |
| **Neutral** | `calm still water, balanced soft lighting, clean minimalist background, steady rhythm, peaceful zen garden, symmetrical composition` |

---

## 4. Image Conditioned (Realistic)
**Method:** ControlNet (Canny) Only
**Logic:** `Base Prompt` + `, {emotion} facial expression, emotional {emotion}`
**Negative Prompt:** `cartoon, anime, 3d render, illustration, painting, low quality, blurry`

*Note: This style relies primarily on the input image structure via ControlNet and does not use elaborate atmospheric descriptions.*
