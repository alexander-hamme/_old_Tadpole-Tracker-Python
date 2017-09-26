from collections import deque
import math


class Animal:

    DEFAULT_BUFFER = 24  # length of line trailing behind each animal

    def __init__(self, point, r, t):
        self.x, self.y = point
        self.radius = int(r)
        self.velocity_vector = (0, 0)
        self.accel_vector = (0, 0)
        self.jerk_vector = 0
        self.current_direction = 0
        self.data_points = [(t, point, (0, 0), (0, 0), 0), ]
        self.line_points = deque(maxlen=self.DEFAULT_BUFFER)

        self.selection_index = 5

    def update_location(self, point, r, t):
        # calculate speed, accel, etc.
        # finally, update coordinates
        self.x, self.y = point

        self.parse_physics(point, t)

        self.data_points.append((t, (self.x, self.y), self.velocity_vector, self.accel_vector, self.current_direction))

        self.line_points.appendleft((int(point[0]), int(point[1])))

    def parse_physics(self, new_pt, t):

        if len(self.data_points) < self.selection_index:
            return None

        prev_points = self.data_points[-self.selection_index:]

        x_points = [pt[1][0] for pt in list(prev_points)] + [new_pt[0]]  # add current coordinates to list
        y_points = [pt[1][1] for pt in list(prev_points)] + [new_pt[1]]

        delta_time = t - prev_points[0][0]

        dx = x_points[-1] - x_points[0]
        dy = y_points[-1] - y_points[0]

        vx = dx / delta_time
        vy = -1 * (dy / delta_time)

        angle = 180/math.pi * math.atan2(vy, vx)

        # Use navigational bearings - straight up is 0 degrees.
        self.current_direction = 90 - angle if angle >= -90 else -(270 + angle)

        time_elapsed = (t - prev_points[0][0]) / 1000.0  # time elapsed between first point and the last (most recent)

        x_distance_traveled = 0
        y_distance_traveled = 0

        for j in range(len(x_points) - 1):
            x_distance_traveled += abs(x_points[j] - x_points[j + 1])

        for k in range(len(y_points) - 1):
            y_distance_traveled += abs(y_points[k] - y_points[k + 1])

        x_displacement = x_distance_traveled
        y_displacement = y_distance_traveled

        # X displacement vector can be positive or negative. Y points are swapped because Y axis goes down.
        dx, dy = x_points[-1] - x_points[0], y_points[0] - y_points[-1]

        if dx < 0:
            x_displacement *= -1  # dx / abs(dx)

        if dy < 0:
            y_displacement *= -1  # dy / abs(dy)

        vx = x_displacement / time_elapsed
        vy = y_displacement / time_elapsed

        velocity_vector = (vx ** 2 + vy ** 2) ** 0.5

        # Velocity value should just be scalar? it doesn't really help if its negative...?
        '''
        if any(
                (
                                vx < 0 and vy < 0,
                                vx < 0 and abs(vx) > 1000 * abs(vy),
                                vy < 0 and abs(vy) > 1000 * abs(vx)
                )
        ):
            velocity_vector *= -1
        '''

        if 1 == 0 and len(self.data_points) > self.selection_index:

            ''' or do acceleration using previous velocity vector for Scalar values??'''

            ''' Although, I guess what we care about IS the accel *in response* to what the velocity is doing,
                because that's what'll tell us if it's a C Start?
            '''

            '''vx_0 = self.data_points[-2][2][0]
            vy_0 = self.data_points[-2][2][1]

            accel_x0 = self.data_points[-2][4][0]
            accel_y0 = self.data_points[-2][4][1]'''

            vx_0 = self.data_points[-1][2][0]
            vy_0 = self.data_points[-1][2][1]

            accel_x0 = self.data_points[-1][4][0]
            accel_y0 = self.data_points[-1][4][1]

            # We only care about the scalar values of velocity? NO, shouldn't be.... ?

            # Alternatives:

            # accel_x = (float(vx)**2 - float(vx_0)**2) / (2*float(x_displacement))
            # accel_y = (float(vy)**2 - float(vy_0)**2) / (2*float(y_displacement))
            # accel_x = (vx - vx_0) / time_elapsed  # (abs(vx) - abs(vx_0)) # Convert time to seconds for bigger decimals
            # accel_y = (vy - vy_0) / time_elapsed       # abs(vy) - abs(vy_0)

            # accel_x = 0 if abs(accel_x) < self.ACCEL_MIN else accel_x
            # accel_y = 0 if abs(accel_y) < self.ACCEL_MIN else accel_y


            accel_x = (x_displacement - (vx_0 * time_elapsed)) / (0.5 * (time_elapsed ** 2))
            accel_y = (y_displacement - (vy_0 * time_elapsed)) / (0.5 * (time_elapsed ** 2))

            acceleration_vector = (accel_x ** 2 + accel_y ** 2) ** 0.5

            # prev_velocity = self.data_points[-1][3]
            # total_disp = (x_displacement**2 + y_displacement**2)**0.5
            # acceleration_vector = (total_disp - (prev_velocity * time_elapsed)) / (0.5 * (time_elapsed**2))
            # (velocity_vector**2 - prev_velocity**2) / (2 * total_disp)

            print(accel_x, accel_y, acceleration_vector)
            # acceleration_vector = 0 if abs(acceleration_vector) < self.ACCEL_MIN else acceleration_vector

            if any(
                    (
                                    accel_x < 0 and accel_y < 0,
                                    accel_x < 0 and abs(accel_x) > 1000 * abs(accel_y),
                                    accel_y < 0 and abs(accel_y) > 1000 * abs(accel_x)
                    )
            ):
                # pass
                acceleration_vector *= -1

            print(acceleration_vector)

            # Jerk Vector = Change in acceleration over time

            jerk_x = (accel_x - accel_x0) / time_elapsed  # 1000.0 *
            jerk_y = (accel_y - accel_y0) / time_elapsed  # 1000.0 *

            # jerk_x = 0 if abs(jerk_x) < self.JERK_MIN else jerk_x
            # jerk_y = 0 if abs(jerk_y) < self.JERK_MIN else jerk_y

            jerk_vector = ((jerk_x ** 2 + jerk_y ** 2) ** 0.5)

            # jerk_vector = 0 if abs(jerk_vector) < self.JERK_MIN else jerk_vector


            if any(
                    (
                                    jerk_x < 0 and jerk_y < 0,
                                    jerk_x < 0 and abs(jerk_x) > 1000 * abs(jerk_y),
                                    jerk_y < 0 and abs(jerk_y) > 1000 * abs(jerk_x)
                    )
            ):
                jerk_vector *= -1

                # print ("{} and {} means jerk vector is {}".format(jerk_x, jerk_y, jerk_vector))
                # elif jerk_x < 0 and abs(jerk_x) > abs(jerk_y):
                #     jerk_vector = -jerk_vector
                # elif jerk_y < 0 and abs(jerk_y) > abs(jerk_x):
                #     jerk_vector = -jerk_vector

                # jerk_vector = ((accel_x**2 + accel_y**2)**0.5 - (accel_x0**2 + accel_y0**2)**0.5

            # impulse_vector = 1 * (vx**2 + vy**2)**0.5 - (vx_0**2 + vy_0**2)**0.5    # (Considers mass to be 1)

            '''
            print("Vx is {}, Vx0 is {}, time is {}, x_displacement is {}".format(vx, vx_0, time_elapsed, x_displacement))
            if not (x_displacement == 0 or y_displacement == 0):
                print("Acceleration1 = {}".format((float(vx)**2 - float(vx_0)**2) / (2*float(abs(x_displacement)))))
            print("Acceleration2 = {}".format((abs(vx) - abs(vx_0)) / time_elapsed))
            print("Acceleration3 = {}\n".format((x_displacement - (abs(vx_0) * time_elapsed))/(0.5*(time_elapsed**2))))

            # accel_x = 1000.0 * (float(vx)**2 - float(vx_0)**2) / (2*float(x_displacement))
            # accel_y = 1000.0 * (float(vy)**2 - float(vy_0)**2) / (2*float(y_displacement))
            # accel_x = 1000.0 * (x_displacement - (vx_0 * time_elapsed))/(0.5*(time_elapsed**2))
            # accel_y = 1000.0 * (y_displacement - (vy_0 * time_elapsed))/(0.5*(time_elapsed**2))
            '''

        # time, position, velocity, velocity vector, acceleration, accel vector, jerk, radius

        self.velocity_vector = (vx, vy)
        self.accel_vector = (0.5, 0.5)
        self.jerk_vector = (0.5, -0.5)
