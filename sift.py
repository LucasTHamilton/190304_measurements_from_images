import PIL.Image
import PIL.ExifTags as TAGS
import numpy as np
from scipy.optimize import least_squares
from scipy import ndimage
import matplotlib.pyplot as plt


class OutOfSensorBoundsError(Exception):
    pass


class Camera(object):
    def __init__(self):
        self.p = None  # Pose
        self.p0 = None
        self.f = None  # Focal Length in Pixels
        self.sensor_size = (0, 0)
        # Throw error if projected coord is out of sensor bounds
        self.error_on_oob = False

    def projective_transform(self, x):
        """
        This function performs the projective transform on generalized coordinates in the camera reference frame.
        """

        x = np.asarray(x)
        # Assume no intensity column
        x0, y0, z0 = x

        # Camera coors to pixel coors
        u = ((x0 / z0) * self.f) + (self.sensor_size[0] // 2)
        v = ((y0 / z0) * self.f) + (self.sensor_size[1] // 2)

        u_min = np.min(u)
        v_min = np.min(v)

        n = len(u)
        u_list = []
        v_list = []
        if self.error_on_oob:
            for i in range(n):
                if (u_min <= u[i] <= self.sensor_size[0]
                        and v_min <= v[i] <= self.sensor_size[1]):
                    u_list.append(u[i])
                    v_list.append(v[i])
                else:
                    raise OutOfSensorBoundsError("Projected coordinate was outside the sensor")
        else:
            for i in range(n):
                u_list.append(u[i])
                v_list.append(v[i])

        u = np.asarray(u_list)
        v = np.asarray(v_list)

        return np.vstack((u, v))

    @staticmethod
    def make_cam_mtx(fi, theta, psi, translation_vec):

        translation_mtx = np.array([[1, 0, 0, -translation_vec[0]],
                                    [0, 1, 0, -translation_vec[1]],
                                    [0, 0, 1, -translation_vec[2]],
                                    [0, 0, 0, 1]])

        # Apply yaw. It represents rotation around camera's z axis
        cos_fi = np.cos(fi)
        sin_fi = np.sin(fi)
        R_yaw = np.array([[cos_fi, -sin_fi, 0, 0],
                          [sin_fi, cos_fi, 0, 0],
                          [0, 0, 1, 0]])

        # Apply pitch. Represents rotation around camera's x axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_pitch = np.array([[1, 0, 0],
                            [0, cos_theta, sin_theta],
                            [0, -sin_theta, cos_theta]])

        # Apply roll. Represents rotation around camera's y axis
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        R_roll = np.array([[cos_psi, 0, -sin_psi],
                           [0, 1, 0],
                           [sin_psi, 0, cos_psi]])

        R_ax_swp = np.array([[1, 0, 0],
                             [0, 0, -1],
                             [0, 1, 0]])

        return np.matmul(R_ax_swp, np.matmul(R_roll, np.matmul(R_pitch, np.matmul(R_yaw, translation_mtx))))

    def rotational_transform(self, X):
        """
        This function performs the translation and rotation from world coordinates into generalized camera coordinates.
        """

        # Unpack pose? could do something different here.
        X_cam, Y_cam, Z_cam, azimuth_cam_deg, pitch_cam_deg, roll_cam_deg = self.p

        # Make X a set of homogeneous coors
        X = np.vstack((X, np.ones(X.shape[1])))

        # Convert degrees to radians
        azimuth_cam_rad = np.deg2rad(azimuth_cam_deg)
        pitch_cam_rad = np.deg2rad(pitch_cam_deg)
        roll_cam_rad = np.deg2rad(roll_cam_deg)

        translation_vec = [X_cam, Y_cam, Z_cam]
        C = self.make_cam_mtx(azimuth_cam_rad, pitch_cam_rad, roll_cam_rad, translation_vec)

        return np.matmul(C, X)

    def estimate_pose(self, X_gcp, u_gcp):
        """
        This function adjusts the pose vector such that the difference between the observed pixel coordinates u_gcp
        and the projected pixels coordinates of X_gcp is minimized.
        """

        self.p = self.p0.copy()

        def residuals(p, x_gcp, u_gcp):
            self.p = p.copy()
            xuv = self.ene_to_camera(x_gcp)
            print(xuv)
            print(u_gcp)
            return xuv.ravel() - u_gcp.ravel()

        res = least_squares(residuals, self.p, args=(X_gcp, u_gcp))
        self.p = res.x

    # f
    def ene_to_camera(self, X):
        return self.projective_transform(self.rotational_transform(X))

    def cam_to_ene(self, u):
        return self.projective_transform(self.rotational_transform(u))


def read_gcp(fname):
    u = []
    v = []
    east = []
    north = []
    ele = []
    with open(fname) as fd:
        for i, line in enumerate(fd):
            if i < 2 or line.startswith("#") or line == "\n":
                # skip header and commented lines
                continue
            vals = line.split(",")
            u.append(float(vals[0]))
            v.append(float(vals[1]))
            east.append(float(vals[2]))
            north.append(float(vals[3]))
            ele.append(float(vals[4]))
    uv = np.vstack((u, v))
    ene = np.vstack((east, north, ele))
    return uv, ene


# Estimates the world coordinates of a specific point in multiple images
def world_coordinate_estimation(camera_list, u_list, ene):
    X_opt = ene

    # Computes the residual: X - guess of world coordinates, c - a camera object, u - pixel coordinates of a point
    def residual(X, camera_list, u_list):
        res_list = np.empty(camera_list.shape[0])
        for c in range(camera_list.shape[0]):
            xuv = camera_list[c].cam_to_ene(X)
            res_list[c] = xuv - u_list[c]

        return res_list

    res = least_squares(residual, X_opt, args=(camera_list, u_list))
    return res


def get_focallength_x_and_y(img):
    with PIL.Image.open(img) as img1:
        exif_data = img1._getexif()

    focal_35 = exif_data[41989]
    try:
        x = exif_data[256]
        y = exif_data[257]
    except:
        x = exif_data[40962]
        y = exif_data[40963]
    focal_length = focal_35/36.0*x
    return focal_length, x, y


if __name__ == "__main__":
    cams = []
    fl, x_res, y_res = get_focallength_x_and_y('campus_stereo_1.jpg')
    sensor_size = (x_res, y_res)
    uv, ene = read_gcp("gcp_stereo_1.txt")
    cam = Camera()
    cam.f = fl
    cam.sensor_size = sensor_size

    # Use one of the points as a basis for the initial guess
    ene0 = ene[:, 0].tolist()
    # Shift back to make sure all points are on the sensor
    ene0[0] -= 1000

    p0 = ene0
    # Assume looking straight East
    p0.extend([90, 0, 0])

    cam.p0 = np.array(p0)
    cam.estimate_pose(ene, uv)
    cams.append(cam)

    uv, ene = read_gcp("gcp_stereo_2.txt")
    u, v = uv

    # Use one of the points as a basis for the initial guess
    ene0 = ene[:, 0].tolist()
    # Shift back to make sure all points are on the sensor
    ene0[0] -= 1000

    p0 = ene0
    # Assume looking straight East
    p0.extend([90, 0, 0])

    fl, x_res, y_res = get_focallength_x_and_y('campus_stereo_2.jpg')
    sensor_size = (x_res, y_res)

    cam = Camera()
    cam.f = fl
    cam.sensor_size = sensor_size
    cam.p0 = np.array(p0)
    cam.estimate_pose(ene, uv)

    # Plot the uv ground control points
    plt.figure(figsize=(10, 13))
    plt.scatter(u, v, color='r')
    plt.show()

    # Visually test the fit
    puv = cam.ene_to_camera(ene)
    plt.figure(figsize=(15, 15))
    plt.scatter(u, v, color='r', s=50, label="GCP")
    plt.scatter(puv[0], puv[1], color='c', s=50, label="Estimate")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
