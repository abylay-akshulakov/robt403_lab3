#!/usr/bin/env python3
# Minimal 3R via-point planner + ROS1 publisher (joints 1,3,5). Joints 2 and 4 locked.
# Python 3.8+, ROS1 Noetic. Requires: numpy, rospy, std_msgs
import argparse, math, sys
import numpy as np

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def ik_3R_xyphi(x, y, phi, l1, l2, l3, x0, y0, qlims):
    xw = (x - x0) - l3*math.cos(phi)
    yw = (y - y0) - l3*math.sin(phi)
    r2 = xw*xw + yw*yw
    c2 = (r2 - l1*l1 - l2*l2) / (2*l1*l2)
    if c2 < -1.0 or c2 > 1.0:
        return np.zeros((0,3))
    s2root = math.sqrt(max(0.0, 1.0 - c2*c2))
    sols = []
    for s2 in (s2root, -s2root):
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(yw, xw) - math.atan2(l2*s2, l1 + l2*c2)
        q3 = (phi - q1 - q2)
        q = wrap_to_pi(np.array([q1, q2, q3], dtype=float))
        if np.all((q >= qlims[:,0]) & (q <= qlims[:,1])):
            sols.append(q)
    return np.vstack(sols) if sols else np.zeros((0,3))

def cubic_coeffs(q0, qf, qd0, qdf, T):
    if T <= 0: raise ValueError("T must be > 0")
    a0 = q0
    a1 = qd0
    a2 = (3*(qf-q0) - (2*qd0+qdf)*T) / (T*T)
    a3 = (2*(q0-qf) + (qd0+qdf)*T) / (T*T*T)
    return np.vstack([a0,a1,a2,a3])

def sample_cubic(A, tt):
    tau = tt.reshape(-1,1)
    a0,a1,a2,a3 = A
    q = a0 + a1*tau + a2*tau**2 + a3*tau**3
    return q  # (N,3)

def plan(vias, l1,l2,l3, x0,y0, qlims, Tseg, Ns):
    M = vias.shape[0]
    if Tseg.shape[0] != M-1: raise ValueError("Tseg needs M-1 values")
    Qvias = np.zeros((M,3), float)
    for i in range(M):
        x,y,ph = vias[i]
        Qc = ik_3R_xyphi(float(x),float(y),float(ph), l1,l2,l3, x0,y0, qlims)
        if Qc.size == 0: raise ValueError(f"Via {i+1} unreachable")
        if i==0:
            k = int(np.argmin(np.linalg.norm(Qc, axis=1)))
        else:
            diffs = wrap_to_pi(Qc - Qvias[i-1])
            k = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        Qvias[i] = Qc[k]

    t_all = []; q_all = []; tcur = 0.0
    for s in range(M-1):
        q0 = Qvias[s].copy()
        qf = Qvias[s+1].copy()
        qd0 = np.zeros(3)
        qdf = np.zeros(3)
        if s>0: qd0 = (Qvias[s+1]-Qvias[s-1])/(Tseg[s-1]+Tseg[s])
        if s<M-2: qdf = (Qvias[s+2]-Qvias[s])/(Tseg[s]+Tseg[s+1])
        T = float(Tseg[s])
        A = cubic_coeffs(q0,qf,qd0,qdf,T)
        tt = np.linspace(0.0, T, int(Ns), dtype=float)
        q = sample_cubic(A, tt)
        t_all.append(tcur + tt)
        q_all.append(q)
        tcur += T
    t_all = np.concatenate(t_all, axis=0)
    q_all  = np.vstack(q_all)
    q_all = np.clip(q_all, qlims[:,0], qlims[:,1])
    return t_all, q_all, Qvias

def publish_ros(t_all, q_all, rate_hz, lock2, lock4, topics, sgn):
    try:
        import rospy
        from std_msgs.msg import Float64
    except Exception as e:
        print("ROS not available:", e); sys.exit(1)

    rospy.init_node("via_traj_player_min", anonymous=True)
    pubs = [rospy.Publisher(t, Float64, queue_size=1) for t in topics]

    dt = 1.0/float(rate_hz)
    t_end = float(t_all[-1])
    t_grid = np.arange(0.0, t_end+1e-9, dt)
    j1 = np.interp(t_grid, t_all, sgn[0]*q_all[:,0])
    j3 = np.interp(t_grid, t_all, sgn[1]*q_all[:,1])
    j5 = np.interp(t_grid, t_all, sgn[2]*q_all[:,2])
    j2 = np.full_like(j1, lock2)
    j4 = np.full_like(j1, lock4)

    try:
        if rospy.get_param("/use_sim_time", False):
            while not rospy.is_shutdown() and rospy.Time.now().to_sec() == 0.0:
                rospy.sleep(0.01)
    except Exception:
        pass

    t0 = rospy.Time.now()
    for k in range(len(t_grid)):
        if rospy.is_shutdown():
            break
        vals = (j1[k], j2[k], j3[k], j4[k], j5[k])
        for pub,v in zip(pubs, vals):
            pub.publish(Float64(float(v)))
        target = t0 + rospy.Duration.from_sec(float(t_grid[k]))
        now = rospy.Time.now()
        if target > now:
            try:
                rospy.sleep((target - now).to_sec())
            except rospy.ROSInterruptException:
                break

    try:
        rospy.loginfo("Trajectory complete.")
    except Exception:
        pass

def parse_vias(s, phi_deg):
    triples = []
    for chunk in s.split(","):
        p = chunk.strip().split()
        if not p: continue
        if len(p)!=3: raise ValueError("via must be x y phi")
        x,y,ph = map(float, p)
        if phi_deg: ph = math.radians(ph)
        triples.append([x,y,ph])
    if not triples: raise ValueError("no via points parsed")
    return np.array(triples, float)

def main():
    ap = argparse.ArgumentParser(description="Minimal 3R via planner + ROS publisher")
    ap.add_argument("--l", default="135,135,46.7")
    ap.add_argument("--base", default="86,0")
    ap.add_argument("--qlims_deg", default="-150 150,-150 150,-150 150")
    ap.add_argument("--vias", default="180 200 0, 300 120 90, 310 -100 0, 180 -200 90")
    ap.add_argument("--phi_deg", action="store_true", default=True)
    ap.add_argument("--Tseg", default="1.5,1.5,1.5")
    ap.add_argument("--Ns", type=int, default=120)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--lock2", type=float, default=0.0)
    ap.add_argument("--lock4", type=float, default=0.0)
    ap.add_argument("--j1", default="/robot/joint1_position_controller/command")
    ap.add_argument("--j2", default="/robot/joint2_position_controller/command")
    ap.add_argument("--j3", default="/robot/joint3_position_controller/command")
    ap.add_argument("--j4", default="/robot/joint4_position_controller/command")
    ap.add_argument("--j5", default="/robot/joint5_position_controller/command")
    ap.add_argument("--sgn1", type=float, default=1.0)
    ap.add_argument("--sgn3", type=float, default=1.0)
    ap.add_argument("--sgn5", type=float, default=1.0)
    args = ap.parse_args()

    l1,l2,l3 = [float(x) for x in args.l.split(",")]
    x0,y0 = [float(x) for x in args.base.split(",")]
    Tseg = np.array([float(x) for x in args.Tseg.split(",")], float)
    vias = parse_vias(args.vias, phi_deg=args.phi_deg)
    qd_pairs = [tuple(map(float, s.strip().split())) for s in args.qlims_deg.split(",")]
    if len(qd_pairs)!=3: sys.exit("qlims_deg must have 3 pairs")
    qlims = np.radians(np.array(qd_pairs, float))  # 3x2

    t_all, q_all, _ = plan(vias, l1,l2,l3, x0,y0, qlims, Tseg=Tseg, Ns=args.Ns)

    topics = [args.j1, args.j2, args.j3, args.j4, args.j5]
    sgn = (args.sgn1, args.sgn3, args.sgn5)
    publish_ros(t_all, q_all, rate_hz=args.rate, lock2=args.lock2, lock4=args.lock4, topics=topics, sgn=sgn)

if __name__ == "__main__":
    main()

