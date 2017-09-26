from collections import deque
import math


class Animal:

    DEFAULT_BUFFER = 24  # length of line trailing behind each animal
    VELOCITY_MULTIPLIER = 1000.0

    def __init__(self, point, r, t):
        self.x, self.y = point
        self.radius = int(r)
        self.velocity_vector = (0, 0)
        self.accel_vector = (0, 0)
        self.jerk_vector = 0
        self.current_direction = 0
        self.line_points = deque(maxlen=self.DEFAULT_BUFFER)
        self.data_points = [(t, point, (0, 0), (0, 0), 0), ]        # time, position, velocity, accel, direction
        self.selection_index = 5                                    # buffer of previous data points to run calculations on

    def update_location(self, point, r, t):
        # calculate speed, accel, etc.
        # finally, update coordinates
        self.x, self.y = point

        self.parse_physics(point, t)

        self.data_points.append((t, (self.x, self.y), self.velocity_vector, self.accel_vector, self.current_direction))

        self.line_points.appendleft((int(point[0]), int(point[1])))

    def parse_physics(self, new_pt, t):

        if len(self.data_points) < self.selection_index:            # don't run calculations until there are enough data points
            return

        prev_points = self.data_points[-self.selection_index:]

        x_points = [pt[1][0] for pt in list(prev_points)] + [new_pt[0]]  # make x and y lists of past coordinates + current coordinate
        y_points = [pt[1][1] for pt in list(prev_points)] + [new_pt[1]]

        delta_time = t - prev_points[0][0]      # time elapsed between current point and the first in list

        dx = x_points[-1] - x_points[0]
        dy = y_points[-1] - y_points[0]

        vx = self.VELOCITY_MULTIPLIER * dx / delta_time
        vy = -self.VELOCITY_MULTIPLIER * dy / delta_time            # y axis must be reversed

        angle = 180/math.pi * math.atan2(vy, vx)

        # Use navigational bearings - straight up is 0 degrees.
        self.current_direction = 90 - angle if angle >= -90 else -(270 + angle)
        
        # Other calculations - need revision
        '''
        if len(self.data_points) > self.selection_index:

            vx_0 = self.data_points[-2][2][0]
            vy_0 = self.data_points[-2][2][1]

            accel_x0 = self.data_points[-2][4][0]
            accel_y0 = self.data_points[-2][4][1]

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

            # accel_x = 1000.0 * (float(vx)**2 - float(vx_0)**2) / (2*float(x_displacement))
            # accel_y = 1000.0 * (float(vy)**2 - float(vy_0)**2) / (2*float(y_displacement))
            # or:
            # accel_x = 1000.0 * (x_displacement - (vx_0 * time_elapsed))/(0.5*(time_elapsed**2))
            # accel_y = 1000.0 * (y_displacement - (vy_0 * time_elapsed))/(0.5*(time_elapsed**2))
            '''

        self.velocity_vector = (vx, vy)
        
        # Not implemented yet
        self.accel_vector = (0.0, 0.0)
        self.jerk_vector = (0.0, 0,0)
