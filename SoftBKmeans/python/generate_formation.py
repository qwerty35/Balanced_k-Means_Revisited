import numpy as np

def generate_letter_positions(letter='A', n_agents=10000, width=1000, height=1000, thickness=10):
    agents = []

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
            agents.append((
                x + np.random.uniform(-thickness, thickness),
                y + np.random.uniform(-thickness, thickness)
            ))

        # 위쪽 반원 (오른쪽으로 열린 반원)
        r_top = width * 0.25
        center_top = (cx, height * 0.75)
        for i in range(n_arc):
            theta = np.pi * (i / n_arc) - np.pi / 2  # [-π/2, π/2]
            x = center_top[0] + r_top * np.cos(theta)
            y = center_top[1] + r_top * np.sin(theta)
            agents.append((
                x + np.random.uniform(-thickness, thickness),
                y + np.random.uniform(-thickness, thickness)
            ))

        # 아래쪽 반원 (오른쪽으로 열린 반원)
        center_bot = (cx, height * 0.25)
        for i in range(n_arc):
            theta = np.pi * (i / n_arc) - np.pi / 2  # [-π/2, π/2]
            x = center_bot[0] + r_top * np.cos(theta)
            y = center_bot[1] + r_top * np.sin(theta)
            agents.append((
                x + np.random.uniform(-thickness, thickness),
                y + np.random.uniform(-thickness, thickness)
            ))

    elif letter.upper() == 'C':
        # C자의 중심
        cx, cy = width / 2, height / 2
        r = min(width, height) * 0.4

        # θ ∈ [π/2, 3π/2]: 왼쪽으로 열린 반원
        for i in range(n_agents):
            theta = np.pi / 2 + (np.pi * i / n_agents)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            agents.append((
                x + np.random.uniform(-thickness, thickness),
                y + np.random.uniform(-thickness, thickness)
            ))

    else:
        raise ValueError("지원하지 않는 문자입니다. 'A', 'B', 'C' 중 하나를 선택하세요.")

    return np.array(agents)

positions = generate_letter_positions(letter='A', n_agents=10000)
np.savetxt("../datasets/s1_generated_Ashape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='B', n_agents=10000)
np.savetxt("../datasets/s1_generated_Bshape.txt", positions.astype(int), fmt="%6d")
positions = generate_letter_positions(letter='C', n_agents=10000)
np.savetxt("../datasets/s1_generated_Cshape.txt", positions.astype(int), fmt="%6d")