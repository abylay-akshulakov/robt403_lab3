#!/usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float64, String

# ---------- math ----------
def wrap_pi(a): 
    return (a + math.pi) % (2*math.pi) - math.pi

def ik_3R(x, y, phi, l1, l2, l3, x0, y0, qlims):
    # orientation-aware IK with EE offset l3
    xw = (x - x0) - l3*math.cos(phi)
    yw = (y - y0) - l3*math.sin(phi)
    r2 = xw*xw + yw*yw
    c2 = (r2 - l1*l1 - l2*l2)/(2*l1*l2)
    if c2 < -1.0 or c2 > 1.0:
        return []
    sols = []
    s2c = [ math.sqrt(max(0.0,1-c2*c2)), -math.sqrt(max(0.0,1-c2*c2)) ]
    for s2 in s2c:
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(yw, xw) - math.atan2(l2*s2, l1 + l2*c2)
        q3 = wrap_pi(phi - q1 - q2)
        q  = (wrap_pi(q1), wrap_pi(q2), q3)
        inside = all(qlims[i][0] <= q[i] <= qlims[i][1] for i in range(3))
        if inside:
            sols.append(q)
    return sols

def angdist(a,b): 
    return wrap_pi(a-b)

def choose_near(solutions, qref):
    return min(solutions, key=lambda q: math.sqrt(sum(angdist(q[i],qref[i])**2 for i in range(3))))

def cubic_rest(q0, qf, T, t):
    # rest-to-rest cubic at time t in [0,T]
    a0 = q0; a1 = 0.0
    a2 =  3*(qf-q0)/(T*T)
    a3 = -2*(qf-q0)/(T*T*T)
    return a0 + a1*t + a2*t*t + a3*t*t*t

# ---------- node ----------
class ThreeRNode:
    def __init__(self):
        rospy.init_node("threeR_cmd_node")

        # geometry (meters, radians)
        self.l1 = rospy.get_param("~l1", 0.135)
        self.l2 = rospy.get_param("~l2", 0.135)
        self.l3 = rospy.get_param("~l3", 0.0467)
        self.x0 = rospy.get_param("~x0", 0.086)
        self.y0 = rospy.get_param("~y0", 0.0)
        lim = float(rospy.get_param("~lim_deg", 90.0)) * math.pi/180.0
        self.qlims = [(-lim, lim)]*3

        # topics
        j1 = rospy.get_param("~j1_topic", "/robot/joint1_position_controller/command")
        j3 = rospy.get_param("~j3_topic", "/robot/joint3_position_controller/command")
        j5 = rospy.get_param("~j5_topic", "/robot/joint5_position_controller/command")
        j2 = rospy.get_param("~j2_topic", "/robot/joint2_position_controller/command")
        j4 = rospy.get_param("~j4_topic", "/robot/joint4_position_controller/command")
        self.lock_j24 = rospy.get_param("~lock_j24", True)

        # pubs
        self.p1 = rospy.Publisher(j1, Float64, queue_size=10)
        self.p3 = rospy.Publisher(j3, Float64, queue_size=10)
        self.p5 = rospy.Publisher(j5, Float64, queue_size=10)
        self.p2 = rospy.Publisher(j2, Float64, queue_size=10) if self.lock_j24 and j2 else None
        self.p4 = rospy.Publisher(j4, Float64, queue_size=10) if self.lock_j24 and j4 else None

        # state
        self.rate_hz = int(rospy.get_param("~rate_hz", 50))
        self.q_now = [0.0, 0.0, 0.0]  # start at "reset"
        rospy.sleep(0.3)

        # cmd subscriber (matches your example pattern)
        rospy.Subscriber("/threeR/cmd", String, self.on_cmd)

        # keep 2 and 4 locked
        rospy.Timer(rospy.Duration(0.2), self._hold_locked)

        rospy.loginfo("Ready. Example: rostopic pub /threeR/cmd std_msgs/String \"reset\" -1")
        rospy.spin()

    # --- helpers ---
    def _hold_locked(self, _evt):
        if self.p2: self.p2.publish(Float64(0.0))
        if self.p4: self.p4.publish(Float64(0.0))

    def _send(self, q):
        # map 3R -> joints 1,3,5
        self.p1.publish(Float64(q[0]))
        self.p3.publish(Float64(q[1]))
        self.p5.publish(Float64(q[2]))
        self._hold_locked(None)

    def _move_to(self, q_goal, T=2.5):
        r = rospy.Rate(self.rate_hz)
        steps = max(1, int(T*self.rate_hz))
        for k in range(steps+1):
            t = T * (k/float(steps))
            q = [cubic_rest(self.q_now[j], q_goal[j], T, t) for j in range(3)]
            self._send(q)
            r.sleep()
        self.q_now = list(q_goal)

    # --- command handler ---
    def on_cmd(self, msg):
        text = msg.data.strip()
        if not text:
            return
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("h","help","?"):
            rospy.loginfo("Commands: reset | goto X_mm Y_mm PHI_deg [T_s]")
            return

        if cmd == "reset":
            self._move_to((0.0,0.0,0.0), T=2.0)
            return

        if cmd == "goto":
            if len(parts) < 4:
                rospy.logwarn("Usage: goto X_mm Y_mm PHI_deg [T_s]")
                return
            x = float(parts[1])/1000.0
            y = float(parts[2])/1000.0
            phi = math.radians(float(parts[3]))
            T = float(parts[4]) if len(parts) >= 5 else 2.5

            sols = ik_3R(x, y, phi, self.l1, self.l2, self.l3, self.x0, self.y0, self.qlims)
            if not sols:
                rospy.logwarn("Unreachable target")
                return
            qf = choose_near(sols, self.q_now)
            self._move_to(qf, T=T)
            return

        rospy.logwarn("Unknown command. Use: help")

if __name__ == "__main__":
    ThreeRNode()
