import numpy as np

def generate_letter_positions(letter='A', n_agents=30000, width=1000, height=1000, thickness=10):
    agents = []

    def add_noise(x, y):
        return (x + np.random.uniform(-thickness, thickness),
                y + np.random.uniform(-thickness, thickness))

    if letter.upper() == 'A':
        # Two legs and a horizontal bar
        n_leg = n_agents // 2
        n_cross = n_agents - 2 * (n_leg // 2)

        # Left leg
        for i in range(n_leg // 2):
            t = i / (n_leg // 2)
            x = (1 - t) * 100 + t * (width // 2)
            y = t * height
            agents.append((x + np.random.uniform(-thickness, thickness),
                           y + np.random.uniform(-thickness, thickness)))

        # Right leg
        for i in range(n_leg // 2):
            t = i / (n_leg // 2)
            x = (1 - t) * (width - 100) + t * (width // 2)
            y = t * height
            agents.append((x + np.random.uniform(-thickness, thickness),
                           y + np.random.uniform(-thickness, thickness)))

        # Horizontal crossbar
        y_cross = height * 0.55
        x_start = width * 0.3
        x_end = width * 0.7
        for i in range(n_cross):
            x = x_start + (x_end - x_start) * (i / n_cross)
            y = y_cross + np.random.uniform(-thickness, thickness)
            agents.append((x + np.random.uniform(-thickness, thickness), y))

    elif letter.upper() == 'B':

        # 비율로 나눔
        n_spine = n_agents // 4
        n_arc = (n_agents - n_spine) // 2
        cx = width * 0.3

        # 세로 기둥 (왼쪽 기둥)
        for i in range(n_spine):
            y = (i / n_spine) * height
            x = cx
            agents.append(add_noise(x, y))

        # 위쪽 반원 (오른쪽으로 열린 반원)
        r_top = width * 0.25
        center_top = (cx, height * 0.75)
        for i in range(n_arc):
            theta = np.pi * (i / n_arc) - np.pi / 2  # [-π/2, π/2]
            x = center_top[0] + r_top * np.cos(theta)
            y = center_top[1] + r_top * np.sin(theta)
            agents.append(add_noise(x, y))

        # 아래쪽 반원 (오른쪽으로 열린 반원)
        center_bot = (cx, height * 0.25)
        for i in range(n_arc):
            theta = np.pi * (i / n_arc) - np.pi / 2  # [-π/2, π/2]
            x = center_bot[0] + r_top * np.cos(theta)
            y = center_bot[1] + r_top * np.sin(theta)
            agents.append(add_noise(x, y))

    elif letter.upper() == 'C':
        # C자의 중심
        cx, cy = width / 2, height / 2
        r = min(width, height) * 0.4

        # θ ∈ [π/2, 3π/2]: 왼쪽으로 열린 반원
        for i in range(n_agents):
            theta = np.pi / 2 + (np.pi * i / n_agents)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            agents.append(add_noise(x, y))

    elif letter.upper() == 'D':
        n_spine = n_agents // 4
        n_arc = n_agents - n_spine
        cx = width * 0.3

        # 세로 기둥 (왼쪽)
        for i in range(n_spine):
            y = (i / n_spine) * height
            x = cx
            agents.append(add_noise(x, y))

        # 오른쪽 반원
        r = height * 0.5
        center = (cx, height / 2)
        for i in range(n_arc):
            theta = np.pi * (i / n_arc) - np.pi / 2  # [-π/2, π/2]
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            agents.append(add_noise(x, y))

    elif letter.upper() == 'E':
        n_vertical = n_agents // 4
        n_horizontal = (n_agents - n_vertical) // 3
        total_assigned = n_vertical + 3 * n_horizontal
        remaining = n_agents - total_assigned

        # 세로 기둥
        for i in range(n_vertical):
            y = (i / (n_vertical - 1)) * height
            x = width * 0.3
            agents.append(add_noise(x, y))

        # 가로 막대 (위, 중간, 아래)
        for frac in [1.0, 0.5, 0.0]:
            y = height * frac
            for i in range(n_horizontal):
                x = width * 0.3 + (width * 0.4) * (i / (n_horizontal - 1))
                agents.append(add_noise(x, y))

        # 남은 agent들 보정 (맨 아래 가로선에 추가)
        for i in range(remaining):
            x = width * 0.3 + (width * 0.4) * (i / max(1, remaining - 1))
            y = 0
            agents.append(add_noise(x, y))

    elif letter.upper() == 'F':
        n_vertical = n_agents // 3
        n_horizontal = n_agents - n_vertical
        n_hbar1 = n_horizontal // 2
        n_hbar2 = n_horizontal - n_hbar1

        for i in range(n_vertical):
            y = (i / (n_vertical - 1)) * height if n_vertical > 1 else height / 2
            x = width * 0.3
            agents.append(add_noise(x, y))

        y1 = height * 1.0
        for i in range(n_hbar1):
            x = width * 0.3 + (width * 0.4) * (i / (n_hbar1 - 1)) if n_hbar1 > 1 else width * 0.5
            agents.append(add_noise(x, y1))

        y2 = height * 0.5
        for i in range(n_hbar2):
            x = width * 0.3 + (width * 0.4) * (i / (n_hbar2 - 1)) if n_hbar2 > 1 else width * 0.5
            agents.append(add_noise(x, y2))

    elif letter.upper() == 'G':
        n_curve = int(n_agents * 0.6)
        n_bar = int(n_agents * 0.2)
        n_down = n_agents - n_curve - n_bar
        cx, cy = width / 2, height / 2
        r = min(width, height) * 0.4

        # C 형태 곡선
        for i in range(n_curve):
            theta = np.pi / 2 + (np.pi * i / (n_curve - 1))  # θ ∈ [π/2, 3π/2]
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            agents.append(add_noise(x, y))

        # 오른쪽 가로 막대 (G의 갈라진 부분)
        bar_y = cy
        bar_start = cx - r * 0.25
        bar_end = cx + r * 0.25
        for i in range(n_bar):
            x = bar_start + (bar_end - bar_start) * (i / n_bar)
            y = bar_y
            agents.append(add_noise(x, y))

        # 아래로 내려가는 짧은 세로 막대 (G의 끝 꼬리)
        vertical_x = bar_end
        for i in range(n_down):
            y = bar_y - (r * i / n_down)
            agents.append(add_noise(vertical_x, y))

    elif letter.upper() == 'H':
        n_vertical = n_agents // 3
        n_cross = n_agents - 2 * n_vertical
        for side in [width * 0.3, width * 0.7]:
            for i in range(n_vertical):
                y = (i / n_vertical) * height
                agents.append(add_noise(side, y))
        y_cross = height * 0.5
        for i in range(n_cross):
            x = width * 0.3 + (width * 0.4) * (i / n_cross)
            agents.append(add_noise(x, y_cross))

    elif letter.upper() == 'I':
        for i in range(n_agents):
            y = (i / n_agents) * height
            x = width * 0.5
            agents.append(add_noise(x, y))

    elif letter.upper() == 'J':
        n_vertical = n_agents // 2
        n_curve = n_agents - n_vertical
        cx = width * 0.5
        r = width * 0.25
        cy = r + 50  # 반원 중심 높이

        # 아래쪽 반원 (시계 방향: 오른쪽 → 왼쪽)
        for i in range(n_curve):
            theta = np.pi + np.pi * i / (n_curve - 1)  # θ ∈ [π, 2π]
            x = cx + r * np.cos(theta) - 0.5 * r
            y = cy + r * np.sin(theta)
            agents.append(add_noise(x, y))

        # 세로 기둥 (반원 위에서 위로 올라감)
        for i in range(n_vertical):
            t = i / (n_vertical - 1)
            y = cy + t * (height - cy - r)
            x = cx + 0.5 * r
            agents.append(add_noise(x, y))

    else:
        raise ValueError("지원하지 않는 문자입니다. 'A~J' 중 하나를 선택하세요.")

    return np.array(agents)

positions = generate_letter_positions(letter='A', n_agents=30)
np.savetxt("../datasets/s3_generated_Ashape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='B', n_agents=30)
np.savetxt("../datasets/s3_generated_Bshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='C', n_agents=30)
np.savetxt("../datasets/s3_generated_Cshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='D', n_agents=30)
np.savetxt("../datasets/s3_generated_Dshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='E', n_agents=30)
np.savetxt("../datasets/s3_generated_Eshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='F', n_agents=30)
np.savetxt("../datasets/s3_generated_Fshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='G', n_agents=30)
np.savetxt("../datasets/s3_generated_Gshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='H', n_agents=30)
np.savetxt("../datasets/s3_generated_Hshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='I', n_agents=30)
np.savetxt("../datasets/s3_generated_Ishape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='J', n_agents=30)
np.savetxt("../datasets/s3_generated_Jshape.txt", positions.astype(int), fmt="%6d")
