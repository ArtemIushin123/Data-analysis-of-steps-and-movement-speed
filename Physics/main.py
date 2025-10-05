import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

df = pd.read_csv('data.txt', delimiter='\t')

df['Time (s)'] = df['Time (s)'] - df['Time (s)'].iloc[0]

accel = df['Absolute acceleration (m/s^2)'].values
time = df['Time (s)'].values

def butter_lowpass_filter(data, cutoff_freq=5, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


filtered_accel = butter_lowpass_filter(accel)

peaks, _ = find_peaks(filtered_accel,
                      height=10,
                      distance=15,
                      prominence=2)


def calculate_velocity(acceleration, time):
    accel_no_gravity = acceleration - 9.8
    velocity = np.zeros_like(accel_no_gravity)
    for i in range(1, len(accel_no_gravity)):
        dt = time[i] - time[i - 1]
        velocity[i] = velocity[i - 1] + accel_no_gravity[i] * dt

    return velocity

velocity = calculate_velocity(accel, time)

def calculate_distance(velocity, time):
    distance = np.zeros_like(velocity)
    for i in range(1, len(velocity)):
        dt = time[i] - time[i - 1]
        distance[i] = distance[i - 1] + abs(velocity[i]) * dt
    return distance


distance = calculate_distance(velocity, time)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time, accel, label='Абсолютное ускорение', alpha=0.7)
plt.plot(time, filtered_accel, label='Фильтрованное ускорение', linewidth=2)
plt.plot(time[peaks], filtered_accel[peaks], 'ro', label=f'Шаги ({len(peaks)} обнаружено)')
plt.xlabel('Время (с)')
plt.ylabel('Ускорение (м/с²)')
plt.title('Обнаружение шагов по ускорению')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, velocity, label='Скорость', color='green')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Скорость перемещения')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, distance, label='Расстояние', color='red')
plt.xlabel('Время (с)')
plt.ylabel('Расстояние (м)')
plt.title('Пройденное расстояние')
plt.grid(True)

plt.tight_layout()
plt.show()

total_steps = len(peaks)
total_distance = distance[-1]
total_time = time[-1]
average_speed = total_distance / total_time if total_time > 0 else 0
step_length = total_distance / total_steps if total_steps > 0 else 0

print("=" * 50)
print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
print("=" * 50)
print(f"Общее количество шагов: {total_steps}")
print(f"Общее время: {total_time:.2f} с")
print(f"Пройденное расстояние: {total_distance:.2f} м")
print(f"Средняя скорость: {average_speed:.2f} м/с ({average_speed * 3.6:.2f} км/ч)")
print(f"Средняя длина шага: {step_length:.2f} м")
print(f"Частота шагов: {total_steps / total_time * 60:.1f} шагов/мин")

print("\n" + "=" * 50)
print("АНАЛИЗ ПО ОСЯМ:")
print("=" * 50)
print(f"Максимальное ускорение по X: {df['Acceleration x (m/s^2)'].max():.2f} м/с²")
print(f"Максимальное ускорение по Y: {df['Acceleration y (m/s^2)'].max():.2f} м/с²")
print(f"Максимальное ускорение по Z: {df['Acceleration z (m/s^2)'].max():.2f} м/с²")

if len(peaks) > 1:
    step_intervals = np.diff(time[peaks])
    avg_step_interval = np.mean(step_intervals)
    step_frequency = 1 / avg_step_interval
    print(f"\nСредний интервал между шагами: {avg_step_interval:.2f} с")
    print(f"Средняя частота шагов: {step_frequency:.2f} Гц")