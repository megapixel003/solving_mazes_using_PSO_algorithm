import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

def af(x):
    return np.tanh(x)
def vd_nn(X, W, V):
    net_h = W.T @ X
    y_h = af(net_h)
    return V.T @ y_h


class Enviroment(object):
    def __init__(self, mapImg: str, end_point=None):
        self.mapImgPath = mapImg
        self.mapImg = pygame.image.load(mapImg)
        self.map_copy = self.mapImg.copy()
        
        # map dims
        self.height = self.mapImg.get_height()
        self.width = self.mapImg.get_width()
        
        # window settings
        pygame.display.set_caption("Simulator Robot")
        self.map = pygame.display.set_mode((self.width, self.height))

        # Điểm đích
        self.end_point = end_point
    
    def draw_map(self):
        self.map.blit(self.mapImg, (0, 0))
        # Tô đỏ điểm đích nếu có
        if self.end_point:
            pygame.draw.circle(self.map, RED, (int(self.end_point[0]), int(self.end_point[1])), 10)

    def draw_sensor(self, x, y, points):
        for i in range(0, len(points)):
            pygame.draw.line(self.map, BLUE, (x, y), points[i])
            pygame.draw.circle(self.map, BLUE, points[i], 5)

    def frame(self, pos, rotation):
        n = 80
        cx, cy = pos
        x_axis = (cx + n * math.cos(rotation), cy + n * math.sin(rotation))
        y_axis = (cx + n * math.cos(rotation + math.pi / 2), cy + n * math.sin(rotation + math.pi / 2))
        pygame.draw.line(self.map, RED, (cx, cy), x_axis, 3)
        pygame.draw.line(self.map, GREEN, (cx, cy), y_axis, 3)


class Robot:
    # Lưu trữ các khu vực ngõ cụt
    dead_end_positions = set()

    def __init__(self, startPos, robotImg, map):
        self.x = startPos[0]
        self.y = startPos[1]
        
        # robot variables
        self.theta = 0
        self.v1 = 0
        self.v2 = 0
        self.l = 8
        self.r = 5
        self.alpha = [0, 2/3*np.pi]
        self.beta = [0, 0]

        # graphics
        self.img = pygame.image.load(robotImg)
        self.img = pygame.transform.scale(self.img, (50, 50))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
        
        # time
        self.dataTime = 0
        self.lastTime = pygame.time.get_ticks()
        self.map = map

        # sensors
        self.distances = [0, 0, 0, 0, 0, 0]
        self.points = []
        self.crash = False
        self.cost_function = 0
        self.it = 0

        # Lịch sử vị trí để phát hiện quay vòng tròn
        self.position_history = []
        self.history_length = 2000  # Tăng để robot có thêm thời gian di chuyển
        self.circle_threshold = 6
        self.circle_penalty = 300  # Giảm hình phạt
        self.is_circling = False

        # Lịch sử khoảng cách để kiểm tra tiến triển
        self.distance_history = []
        self.distance_history_length = 1000  # Tăng để robot có thêm thời gian
        self.progress_threshold = 0.05
        self.no_progress_penalty = 300  # Giảm hình phạt
        self.no_progress = False
        self.no_progress_count = 0

        # Đường đi và vị trí đã đi qua
        self.path = [(self.x, self.y)]
        self.visited_positions = {(int(self.x), int(self.y))}
        self.previous_distance = float('inf')
        self.previous_position = (self.x, self.y)  # Lưu vị trí trước đó để tính khoảng cách di chuyển

        # Hình phạt khi chạy hết thời gian
        self.timeout_penalty = 300
        self.is_timeout = False

        # Trạng thái đạt mục tiêu
        self.reached_goal = False

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self):
        self.dataTime = (pygame.time.get_ticks() - self.lastTime) / 1000
        self.lastTime = pygame.time.get_ticks()

        R = np.array([
            [np.cos(self.theta), np.sin(self.theta), 0],
            [-np.sin(self.theta), np.cos(self.theta), 0],
            [0, 0, 1]
        ])
        j1f = np.array([
            [np.sin(self.alpha[0] + self.beta[0]), -np.cos(self.alpha[0] + self.beta[0]), -self.l*np.cos(self.beta[0])],
            [np.sin(self.alpha[1] + self.beta[1]), -np.cos(self.alpha[1] + self.beta[1]), -self.l*np.cos(self.beta[1])]
        ])
        j2 = np.array([
            [self.r, 0],
            [0, self.r]
        ])
        v = np.array([
            [self.v1],
            [self.v2]
        ])
        vv = np.linalg.inv(R) @ np.linalg.pinv(j1f) @ j2 @ v
        vv = vv.flatten()
        self.x += vv[0] * self.dataTime
        self.y += vv[1] * self.dataTime
        self.theta += vv[2] * self.dataTime
        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0
        
        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(-self.theta), 1)
        self.rect = self.rotated.get_rect(center = (self.x, self.y))

        self.path.append((self.x, self.y))

    def update_sensor_data(self):
        angles = [
            self.theta,
            np.pi/3 + self.theta,
            2/3 * np.pi + self.theta,
            np.pi + self.theta,
            4/3 * np.pi + self.theta,
            5/3 * np.pi + self.theta
        ]
        distances = []
        points = []

        for angle in angles:
            distance = 0
            x = int(self.x)
            y = int(self.y)
            while True:
                x = int(self.x + distance * np.cos(angle))
                y = int(self.y + distance * np.sin(angle))
                distance += 1
                if x < 0 or x >= self.map.get_width() or y < 0 or y >= self.map.get_height():
                    break
                r, g, b, a = self.map.get_at((x, y))
                if r < 20 and g < 20 and b < 20:
                    break
            distances.append(distance)
            points.append((x, y))
        self.distances = distances
        self.points = points

    def check_crash(self, end_point):
        x, y = int(self.x), int(self.y)
        if x < 0 or x >= self.map.get_width() or y < 0 or y >= self.map.get_height():
            self.crash = True
            self.dead_end_positions.add((int(self.x), int(self.y)))
            self.cost_function = (self.cost_function / self.it) + 300
            return True
        
        r, g, b, a = self.map.get_at((x, y))
        if r < 20 and g < 20 and b < 20:
            self.crash = True
            self.dead_end_positions.add((int(self.x), int(self.y)))
            self.cost_function = (self.cost_function / self.it) + 300
            return True

        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)

        if len(self.position_history) == self.history_length:
            avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
            avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
            max_distance = max(np.sqrt((pos[0] - avg_x)**2 + (pos[1] - avg_y)**2) for pos in self.position_history)
            if max_distance < self.circle_threshold:
                self.is_circling = True
                self.cost_function = (self.cost_function / self.it) + self.circle_penalty
                self.dead_end_positions.add((int(self.x), int(self.y)))
                return True

        distance_to_goal = np.sqrt((self.x - end_point[0])**2 + (self.y - end_point[1])**2)
        self.distance_history.append(distance_to_goal)
        if len(self.distance_history) > self.distance_history_length:
            self.distance_history.pop(0)

        if len(self.distance_history) == self.distance_history_length:
            distance_change = self.distance_history[0] - self.distance_history[-1]
            if abs(distance_change) < self.progress_threshold:
                self.no_progress = True
                self.no_progress_count += 1
                penalty = self.no_progress_penalty + 50 * self.no_progress_count
                self.cost_function = (self.cost_function / self.it) + penalty
                self.dead_end_positions.add((int(self.x), int(self.y)))
                return True
            else:
                self.no_progress_count = 0

        if distance_to_goal < 5:
            self.reached_goal = True
            return True

        self.crash = False
        self.is_circling = False
        self.no_progress = False
        self.is_timeout = False
        self.reached_goal = False
        return False
    
    def apply_timeout_penalty(self):
        if not (self.crash or self.is_circling or self.no_progress or self.is_timeout or self.reached_goal):
            self.is_timeout = True
            self.cost_function = (self.cost_function / self.it) + self.timeout_penalty
            return True
        return False




if __name__ == "__main__":
    pygame.init()

    # start position
    start = (225, 25, np.pi/2)
    end_point = (225, 420, np.pi/2)
    running = True
    run_time = 0
    last_time = pygame.time.get_ticks()
    max_run_time = 600

    # enviroment
    env = Enviroment('./map/map-3.png', end_point)

    robots = []
    num_robot = 20

    # NN
    n_inputs = 12
    n_hidden = 10
    n_outputs = 2

    # Số tham số cần tối ưu bằng PSO
    npar = (n_inputs * n_hidden) + (n_hidden * n_outputs)
    
    pop_size = num_robot
    min_max = [-5, 5]
    w_pso = 0.8
    c1_pso = 1.2
    c2_pso = 1.8
    max_iteration = 50
    current_iteration = 0

    Pbest_fitness = 99999999*np.ones(pop_size)
    Gbest_fitness = 99999999
    Pbest_position = np.zeros((pop_size, npar))
    Gbest_position = np.zeros((npar, 1))
    fitness_values = np.zeros(max_iteration)
    fitness_mean = np.zeros(max_iteration)
    JJ = np.zeros(pop_size)

    P_pso = []
    # lấy trọng số đầu từ file json
    with open("./results/map-3/result13/best_path_and_parameters.json", "r") as f:
        d = json.load(f).get("best_parameters")
        for i in range(pop_size):
            noise = np.random.uniform(-1, 1, len(d))
            P_pso.append(np.array(d) + noise)
        P_pso = np.array(P_pso)
        print("Load parameters from file")
        print(P_pso)
    # Khởi tạo PSO random
    if len(P_pso) == 0:
        P_pso = np.random.uniform(min_max[0], min_max[1], (pop_size, npar))

    V_pso = P_pso * 0

    # Lưu đường đi tốt nhất
    best_path = None
    best_fitness = float('inf')
    best_parameters = None

    try:
        while running and current_iteration < max_iteration:
            num_robot_available = num_robot
            robots = []
            
            for i in range(num_robot):
                robots.append(Robot(start, './robotPenguin.png', env.map_copy))

            run_time = 0
            last_time = pygame.time.get_ticks()
            for robot in robots:
                robot.x = start[0]
                robot.y = start[1]
                robot.theta = start[2]
                robot.v1 = 0
                robot.v2 = 0
                robot.cost_function = 0
                robot.crash = False
                robot.is_circling = False
                robot.no_progress = False
                robot.is_timeout = False
                robot.reached_goal = False
                robot.it = 0
                robot.check_crash(end_point)
                robot.update_sensor_data()

            while num_robot_available > 0 and run_time < max_run_time:
                for idx, robot in enumerate(robots):
                    if not (robot.crash or robot.is_circling or robot.no_progress or robot.is_timeout or robot.reached_goal):
                        robot.update_sensor_data()

                        # Tính toán cost function
                        distance_to_goal = np.sqrt((robot.x - end_point[0])**2 + (robot.y - end_point[1])**2)
                        distance_reward = -2000 / (distance_to_goal + 1)
                        orientation_error = abs(robot.theta - end_point[2])
                        wall_proximity = sum(max(0, 15 - d) for d in robot.distances)
                        step_penalty = 0.01 * robot.it

                        orientation_penalty = abs(robot.theta - np.arctan2(end_point[1] - robot.y, end_point[0] - robot.x))

                        # Thưởng đường đi đột phá
                        breakthrough_reward = 0
                        current_pos = (int(robot.x), int(robot.y))
                        min_dist_to_visited = float('inf')
                        for visited_pos in robot.visited_positions:
                            dist = np.sqrt((current_pos[0] - visited_pos[0])**2 + (current_pos[1] - visited_pos[1])**2)
                            min_dist_to_visited = min(min_dist_to_visited, dist)
                        if min_dist_to_visited > 10:
                            breakthrough_reward = -200

                        # phạt khi quay lại đường cũ
                        old_path_penalty = 0
                        if current_pos in robot.visited_positions:
                            old_path_penalty = 200

                        # Thưởng dựa trên khoảng cách di chuyển
                        movement_reward = 0
                        prev_x, prev_y = robot.previous_position
                        movement_distance = np.sqrt((robot.x - prev_x)**2 + (robot.y - prev_y)**2)
                        if movement_distance > 2:
                            movement_reward = -50 * movement_distance
                        robot.previous_position = (robot.x, robot.y)

                        # Phạt khi quay lại khu vực ngõ cụt
                        dead_end_penalty = 0
                        if current_pos in robot.dead_end_positions:
                            dead_end_penalty = 200

                        # Thưởng đi đường dài
                        total_path_reward = 0
                        if len(robot.path) > 1:
                            for i in range(len(robot.path) - 1):
                                x1, y1 = robot.path[i]
                                x2, y2 = robot.path[i + 1]
                                path_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                if path_length > 10:
                                    total_path_reward += -0.5 * path_length

                        # Tổng hợp cost function
                        robot.cost_function = (
                                              2.0 * breakthrough_reward +
                                              1.5 * distance_reward +
                                              0.8 * distance_to_goal +
                                              1.5 * movement_reward +
                                              1.0 * dead_end_penalty +
                                              0.05 * wall_proximity +
                                              0.02 * orientation_error +
                                              0.01 * step_penalty +
                                              1.0 * old_path_penalty +
                                              0.5 * total_path_reward +
                                              0.2 * orientation_penalty
                                            )
                        
                        robot.it += 1
                        robot.visited_positions.add(current_pos)
                        
                        # Đầu vào NN
                        X_nn = np.array([
                            robot.distances[0], robot.distances[1], robot.distances[2],
                            robot.distances[3], robot.distances[4], robot.distances[5],
                            robot.x, robot.y, robot.theta,
                            end_point[0], end_point[1], end_point[2]
                        ])
                        
                        # Cập nhật trọng số
                        W = P_pso[idx, :(n_inputs * n_hidden)].reshape(n_inputs, n_hidden)
                        V = P_pso[idx, (n_inputs * n_hidden):].reshape(n_hidden, n_outputs)
                        VVV = vd_nn(X_nn, W, V)
                        v1, v2 = VVV.flatten()
                        # Điều chỉnh vận tốc động
                        min_distance = min(robot.distances)
                        speed_factor = 1.0 if min_distance > 20 else min_distance / 20  # Giảm tốc độ khi gần tường
                        robot.v1 = v1 * speed_factor
                        robot.v2 = v2 * speed_factor
                        robot.move()
                        
                        # Kiểm tra va chạm, quay vòng, tiến triển, và đạt mục tiêu
                        if robot.check_crash(end_point):
                            num_robot_available -= 1
                        
                        robot.draw(env.map)
                        env.draw_sensor(robot.x, robot.y, robot.points)
                        env.frame((robot.x, robot.y), robot.theta)
                pygame.display.update()
                env.draw_map()
                run_time = (pygame.time.get_ticks() - last_time) / 1000

            # Áp dụng hình phạt cho các robot còn hoạt động khi hết thời gian
            if run_time >= max_run_time:
                for robot in robots:
                    if robot.apply_timeout_penalty():
                        num_robot_available -= 1

            # Cập nhật PSO và lưu đường đi tốt nhất
            for idx, robot in enumerate(robots):
                J = robot.cost_function
                JJ[idx] = J
                if J < Pbest_fitness[idx]:
                    Pbest_fitness[idx] = J
                    Pbest_position[idx] = P_pso[idx]
                if J < Gbest_fitness:
                    Gbest_fitness = J
                    Gbest_position = P_pso[idx]
                    best_path = robot.path
                    best_fitness = J
                    best_parameters = P_pso[idx]

            fitness_values[current_iteration] = Gbest_fitness
            fitness_mean[current_iteration] = np.mean(JJ)
            
            V_pso = w_pso * V_pso + c1_pso * np.random.rand() * (Pbest_position - P_pso) + c2_pso * np.random.rand() * (Gbest_position - P_pso)
            P_pso = P_pso + V_pso
            print(f'Iteration {current_iteration}: {np.min(JJ)} - {Gbest_fitness}')
            current_iteration += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if best_path:
            path_length = 0
            for i in range(len(best_path) - 1):
                x1, y1 = best_path[i]
                x2, y2 = best_path[i + 1]
                path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            final_pos = best_path[-1]
            final_distance = np.sqrt((final_pos[0] - end_point[0])**2 + (final_pos[1] - end_point[1])**2)
            num_steps = len(best_path) - 1

            data = {
                "time": time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()),
                "best_fitness": float(best_fitness),
                "path_length": float(path_length),
                "number_of_steps": num_steps,
                "final_distance_to_goal": float(final_distance),
                "best_path": [[float(pos[0]), float(pos[1])] for pos in best_path],
                "best_parameters": [float(param) for param in best_parameters]
            }
            print(data)
            
            # Lưu đè lên file JSON nếu fitness tốt hơn
            try:
                with open("best_path_and_parameters.json", "r") as f:
                    existing_data = json.load(f)
                if data["best_fitness"] < existing_data["best_fitness"]:
                    with open("best_path_and_parameters.json", "w") as f:
                        json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Error reading or writing JSON file: {e}")
                with open("best_path_and_parameters.json", "w") as f:
                    json.dump(data, f, indent=4)

            # lưu vào folder: result
            result_i = 0
            result_map_name = env.mapImgPath.split('/')[-1].split('.')[0]
            if not os.path.exists(f'results/{result_map_name}'):
                os.makedirs(f'results/{result_map_name}')
            while os.path.exists(f'results/{result_map_name}/result{result_i}'):
                result_i += 1
            os.makedirs(f'results/{result_map_name}/result{result_i}')
            with open(f'results/{result_map_name}/result{result_i}/best_path_and_parameters.json', "w") as f:
                json.dump(data, f, indent=4)
            # Vẽ đường đi tốt nhất và lưu
            env.map = env.map_copy
            env.draw_map()
            for pos in best_path:
                pygame.draw.circle(env.map_copy, GREEN, (int(pos[0]), int(pos[1])), 5)
            pygame.draw.circle(env.map_copy, GREEN, (int(best_path[0][0]), int(best_path[0][1])), 10)
            pygame.draw.circle(env.map_copy, RED, (int(best_path[-1][0]), int(best_path[-1][1])), 10)
            pygame.image.save(env.map_copy, f'results/{result_map_name}/result{result_i}/best_path.png')

        plt.figure(figsize=(10, 5))
        plt.plot(range(max_iteration), fitness_values, label='Gbest Fitness', color='blue')
        plt.plot(range(max_iteration), fitness_mean, label='Mean Fitness', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()