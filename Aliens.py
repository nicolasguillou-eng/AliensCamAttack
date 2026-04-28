import cv2
import numpy as np
import random
import time
import os

# --- RÉGLAGES ---
WIDTH, HEIGHT = 640, 480
N_PIXELS_ATTAQUE = 300 
SEUIL_PRESENCE = 45     
SEUIL_FRAPPE = 25       
INERTIE = 0.2
GRAVITY = 0.25 
HS_FILE = "Highscores.txt"
WIN_NAME = "Alien Defense Particles"
SCALE_FACTOR = 1.0  

# --- RÉGLAGES ANIMATION PROPORTIONNELLE ---
WAVE_INTERVAL = 10    
SYMMETRIC_ANIM_SPEED = 0.15  
SYMMETRIC_ANIM_AMP_PCT = 0.12 

# --- GESTION DES SCORES ---
def load_highscores():
    scores = []
    if os.path.exists(HS_FILE):
        with open(HS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    scores.append((parts[0], int(parts[1])))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:10]

def save_highscores(scores):
    with open(HS_FILE, 'w', encoding='utf-8') as f:
        for name, score in scores:
            f.write(f"{name},{score}\n")

def display_highscores_screen(image, scores, title="CLASSEMENT TOP 10"):
    overlay = image.copy()
    cv2.rectangle(overlay, (80, 40), (WIDTH-80, HEIGHT-40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, title, (WIDTH//2 - 130, 80), 1, 2, (0, 255, 255), 2)
    if not scores:
        cv2.putText(image, "Aucun score enregistre", (150, HEIGHT//2), 1, 1.2, (200, 200, 200), 1)
    else:
        for i, (name, s) in enumerate(scores):
            text = f"{i+1}. {name[:10]:<10} : {s}"
            cv2.putText(image, text, (150, 130 + i*28), 1, 1.2, (255, 255, 255), 1)
    cv2.putText(image, "Appuyez sur une touche pour continuer", (110, HEIGHT-60), 1, 1.0, (0, 255, 0), 1)
    cv2.imshow(WIN_NAME, image)
    cv2.waitKey(0)

def get_player_name_cv(bg_image, final_score):
    name = ""
    while True:
        temp_img = bg_image.copy()
        cv2.rectangle(temp_img, (100, HEIGHT//2 - 60), (WIDTH-100, HEIGHT//2 + 60), (0, 0, 0), -1)
        cv2.rectangle(temp_img, (100, HEIGHT//2 - 60), (WIDTH-100, HEIGHT//2 + 60), (0, 255, 255), 2)
        cv2.putText(temp_img, "NOUVEAU RECORD !", (WIDTH//2 - 120, HEIGHT//2 - 80), 1, 1.5, (0, 255, 255), 2)
        cv2.putText(temp_img, f"Score: {final_score}", (WIDTH//2 - 60, HEIGHT//2 - 30), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(temp_img, f"NOM: {name}_", (130, HEIGHT//2 + 20), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(temp_img, "Appuyez sur Entree pour valider", (WIDTH//2 - 140, HEIGHT//2 + 50), 1, 0.9, (0, 255, 0), 1)
        cv2.imshow(WIN_NAME, temp_img)
        key = cv2.waitKey(0) & 0xFF
        if key == 13: break
        elif key == 8: name = name[:-1]
        elif len(name) < 10 and (str.isalnum(chr(key)) or key == 32): name += chr(key)
    return name if name.strip() != "" else "Anonyme"

def load_and_slice_aliens():
    fname = 'Aliens.jpg' if os.path.exists('Aliens.jpg') else 'Aliens.png'
    if not os.path.exists(fname): return []
    aliens_img = cv2.imread(fname)
    gray = cv2.cvtColor(aliens_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sprite_list = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        x, y, w, h = cv2.boundingRect(cnt)
        sprite_white = aliens_img[y:y+h, x:x+w].copy()
        sprite_gray = gray[y:y+h, x:x+w]
        _, alpha = cv2.threshold(sprite_gray, 200, 255, cv2.THRESH_BINARY)
        sprite_bgra = cv2.merge([sprite_white[:,:,0], sprite_white[:,:,1], sprite_white[:,:,2], alpha])
        sprite_list.append(sprite_bgra)
    return sprite_list

def load_assets():
    if os.path.exists('background.jpg'):
        bg_img = cv2.resize(cv2.imread('background.jpg'), (WIDTH, HEIGHT))
    elif os.path.exists('background.png'):
        bg_img = cv2.resize(cv2.imread('background.png'), (WIDTH, HEIGHT))
    else:
        bg_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        bg_img[:] = (15, 10, 25)
    return bg_img, load_and_slice_aliens()

def get_animated_sprite_proportional(sprite_bgra, t):
    # Ajout d'une marge (padding) pour éviter de couper le sprite lors de la déformation
    pad = 20
    sprite_padded = cv2.copyMakeBorder(sprite_bgra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0,0])
    
    h, w = sprite_padded.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    ref_size = max(w, h)
    amplitude = ref_size * SYMMETRIC_ANIM_AMP_PCT
    shift = amplitude * np.sin(t)
    
    # Déformation ondulatoire simple (sans symétrie)
    map_x += shift * np.sin((10.0 / h) * map_y)
    
    res = cv2.remap(sprite_padded, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res

def overlay_colored_sprite(background, sprite_bgra, x, y, angle=0, color_bgr=(255, 255, 255)):
    h, w = sprite_bgra.shape[:2]
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.abs(np.cos(angle_rad)), np.abs(np.sin(angle_rad))
    new_w, new_h = int((h * sin_a) + (w * cos_a)), int((h * cos_a) + (w * sin_a))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    alpha = sprite_bgra[:, :, 3] / 255.0
    colored_img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    colored_sprite_rgb = (colored_img * alpha[:, :, np.newaxis]).astype(np.uint8)
    rotated_rgb = cv2.warpAffine(colored_sprite_rgb, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    rotated_alpha = cv2.warpAffine(alpha, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    y1, y2 = max(0, y - new_h//2), min(HEIGHT, y + new_h//2)
    x1, x2 = max(0, x - new_w//2), min(WIDTH, x + new_w//2)
    sy1, sx1 = max(0, new_h//2 - (y - y1)), max(0, new_w//2 - (x - x1))
    sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
    if y2 <= y1 or x2 <= x1: return background
    bg_chunk = background[y1:y2, x1:x2]
    sprite_rgb_chunk, alpha_chunk = rotated_rgb[sy1:sy2, sx1:sx2], rotated_alpha[sy1:sy2, sx1:sx2]
    for c in range(3):
        bg_chunk[:, :, c] = (alpha_chunk * sprite_rgb_chunk[:, :, c] + (1.0 - alpha_chunk) * bg_chunk[:, :, c])
    background[y1:y2, x1:x2] = bg_chunk
    return background

class Particle:
    def __init__(self, x, y, color):
        self.pos = [float(x), float(y)]
        angle, speed = random.uniform(0, 2 * np.pi), random.uniform(2, 7)
        self.vel = [np.cos(angle) * speed, np.sin(angle) * speed]
        self.life, self.decay = 1.0, random.uniform(0.03, 0.07)
        self.color, self.size = color, random.randint(2, 5)
    def update(self):
        self.vel[1] += GRAVITY
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= self.decay
        return self.life > 0

class Alien:
    def __init__(self, speed, source_sprite, player_pos):
        valid_pos = False
        while not valid_pos:
            side = random.choice(['t', 'b', 'l', 'r'])
            if side == 't': temp_pos = [random.randint(0, WIDTH), -50]
            elif side == 'b': temp_pos = [random.randint(0, WIDTH), HEIGHT+50]
            elif side == 'l': temp_pos = [-50, random.randint(0, HEIGHT)]
            else: temp_pos = [WIDTH+50, random.randint(0, HEIGHT)]
            dist_to_player = np.hypot(temp_pos[0] - player_pos[0], temp_pos[1] - player_pos[1])
            if dist_to_player > 200:
                self.pos = temp_pos
                valid_pos = True
        self.source_sprite = source_sprite
        
        # Chronométrage asynchrone
        self.anim_t = random.uniform(0, np.pi * 2) 
        self.anim_phase = 0.0 
        
        r = random.random()
        base_scale = 0.3 + (r**2) * (1.2 - 0.3) 
        self.scale = base_scale * SCALE_FACTOR
        self.speed = speed + (1.5 / self.scale) * random.uniform(0.1, 0.5)
        self.pixel_count = cv2.countNonZero(self.source_sprite[:, :, 3])
        self.angle, self.rot_speed = random.randint(0, 360), random.uniform(2, 6)
        
        # Couleurs vives HSV
        h_val = random.randint(0, 179)
        hsv_img = np.uint8([[[h_val, 255, 255]]])
        bgr_col = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)[0][0]
        self.color = (int(bgr_col[0]), int(bgr_col[1]), int(bgr_col[2]))

        self.drift, self.drift_speed, self.drift_amp = 0.0, random.uniform(0.05, 0.15), random.uniform(0.5, 1.2)

    def move(self, tx, ty):
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        dist = np.hypot(dx, dy)
        if dist > 5:
            base_angle = np.arctan2(dy, dx)
            self.drift += self.drift_speed
            final_angle = base_angle + np.sin(self.drift) * self.drift_amp
            self.pos[0] += np.cos(final_angle) * self.speed
            self.pos[1] += np.sin(final_angle) * self.speed
        self.angle = (self.angle + self.rot_speed) % 360
        
        # Cycle 1s animé / 1s fixe au point zéro
        cycle_pos = self.anim_t % (np.pi * 2)
        if cycle_pos < np.pi:
            self.anim_phase += SYMMETRIC_ANIM_SPEED * 2
        else:
            # Revenir doucement vers le repos (multiple de PI)
            target = np.round(self.anim_phase / np.pi) * np.pi
            self.anim_phase += (target - self.anim_phase) * 0.15
            
        self.anim_t += 0.08

    def draw(self, display):
        h, w = self.source_sprite.shape[:2]
        new_w, new_h = max(1, int(w*self.scale)), max(1, int(h*self.scale))
        resized_static = cv2.resize(self.source_sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Animation sans symétrie
        animated = get_animated_sprite_proportional(resized_static, self.anim_phase)
        return overlay_colored_sprite(display, animated, int(self.pos[0]), int(self.pos[1]), self.angle, self.color)

# --- INIT ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, WIDTH); cap.set(4, HEIGHT)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, WIDTH, HEIGHT)
game_bg_image, source_sprites = load_assets()
if not source_sprites: exit("Besoin de Aliens.jpg")

scores_list = load_highscores()
display_highscores_screen(game_bg_image.copy(), scores_list, "MEILLEURS SCORES")

print("Calibrage...")
time.sleep(1)
for _ in range(15): cap.read()
ret, frame_init = cap.read()
bg_ref = cv2.flip(cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY), 1)
prev_gray = bg_ref.copy()

aliens, particles, flash_effects, points_texts = [], [], [], []
lives, score = 10, 0
p_center = [WIDTH // 2, HEIGHT // 2]
last_wave_time = time.time()
bonus_vagues = 0 

while lives > 0:
    if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_p = cv2.absdiff(cv2.GaussianBlur(bg_ref, (15,15), 0), cv2.GaussianBlur(gray, (15,15), 0))
    _, player_mask = cv2.threshold(diff_p, SEUIL_PRESENCE, 255, cv2.THRESH_BINARY)
    diff_v = cv2.absdiff(prev_gray, gray)
    _, att_t = cv2.threshold(diff_v, SEUIL_FRAPPE, 255, cv2.THRESH_BINARY)
    attack_mask = np.zeros_like(att_t)
    cnts, _ = cv2.findContours(att_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > N_PIXELS_ATTAQUE: cv2.drawContours(attack_mask, [c], -1, 255, -1)
    prev_gray = gray.copy()
    cv2.accumulateWeighted(gray, bg_ref.astype(float), 0.005)

    M = cv2.moments(player_mask)
    if M["m00"] > 1000:
        p_center[0] = int(p_center[0] + (M["m10"]/M["m00"] - p_center[0]) * INERTIE)
        p_center[1] = int(p_center[1] + (M["m01"]/M["m00"] - p_center[1]) * INERTIE)

    display = cv2.add(cv2.bitwise_and(frame, frame, mask=player_mask),
                      cv2.bitwise_and(game_bg_image, game_bg_image, mask=cv2.bitwise_not(player_mask)))
    
    current_time = time.time()
    if current_time - last_wave_time > WAVE_INTERVAL:
        bonus_vagues += 1
        aliens.append(Alien(2.0 + (bonus_vagues * 0.1), random.choice(source_sprites), p_center))
        last_wave_time = current_time

    if len(aliens) < (1 + (score // 10000) + bonus_vagues) and random.random() < 0.06:
        aliens.append(Alien(2.0 + (bonus_vagues * 0.1), random.choice(source_sprites), p_center))
    
    for a in aliens[:]:
        a.move(p_center[0], p_center[1])
        ax, ay = int(a.pos[0]), int(a.pos[1])
        if 0 <= ay < HEIGHT and 0 <= ax < WIDTH:
            if player_mask[ay, ax] == 255:
                if attack_mask[ay, ax] == 255:
                    pts = a.pixel_count * 10
                    score += pts
                    points_texts.append({'text': f"+{pts}", 'pos': (ax, ay), 'color': a.color, 'life': 30})
                    for _ in range(12): particles.append(Particle(ax, ay, a.color))
                    aliens.remove(a)
                else:
                    lives -= 1
                    flash_effects.append([ax, ay, 10, (0,0,255)])
                    aliens.remove(a)
            else:
                display = a.draw(display) 
        elif abs(ax) > WIDTH + 150 or abs(ay) > HEIGHT + 150: aliens.remove(a)

    for p in particles[:]:
        if p.update():
            px, py = int(p.pos[0]), int(p.pos[1])
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                cv2.rectangle(display, (px, py), (px + p.size, py + p.size), p.color, -1)
        else: particles.remove(p)

    for f in flash_effects[:]:
        fx, fy, flife, fcol = f
        ov = display.copy()
        cv2.circle(ov, (fx, fy), (11-flife)*20, fcol, -1)
        cv2.addWeighted(ov, 0.4, display, 0.6, 0, display)
        f[2] -= 1
        if f[2] <= 0: flash_effects.remove(f)
        
    for pt in points_texts[:]:
        tx, ty = pt['pos']
        cv2.putText(display, pt['text'], (tx, ty), 1, 1.2, pt['color'], 2)
        pt['pos'] = (tx, ty - 1); pt['life'] -= 1
        if pt['life'] <= 0: points_texts.remove(pt)

    cv2.putText(display, f"VIES: {lives}  SCORE: {score}  LVL: {bonus_vagues+1}", (20, 40), 1, 1.5, (255,255,255), 2)
    cv2.imshow(WIN_NAME, display)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()