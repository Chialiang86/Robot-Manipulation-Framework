

    # # joint symbols
    # q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
    # joint_angles = [q1, q2, q3, q4, q5, q6, q7] 

    # dh_params = get_panda_DH_params()
    
    # # compute current forward kinematic
    # FK = eye(4)
    # for i, (dh, q) in enumerate(zip(reversed(dh_params), reversed(joint_angles))):
    #     d = dh['d']
    #     a = dh['a']
    #     alpha = dh['alpha']

    #     ca = cos(alpha)
    #     sa = sin(alpha)
    #     cq = cos(q)
    #     sq = sin(q)

    #     transform = Matrix(
    #         [
    #             [cq, -sq, 0, a],
    #             [ca * sq, ca * cq, -sa, -d * sa],
    #             [sa * sq, cq * sa, ca, d * ca],
    #             [0, 0, 0, 1],
    #         ]
    #     )
    #     FK = transform @ FK

    # A = FK[0:3, 0:4]  # crop last row
    # A = A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

    # print('step 1...')
    # Q = Matrix(joint_angles)
    # J = A.jacobian(Q)  # compute Jacobian symbolically
    # print(J)

    # print('step 2...')
    # A_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy'))
    # J_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy'))

    # print('step 3...')
    # q_init = np.array(joint_pose).reshape(7, 1)
    # A_init = A_lamb(*(q_init.flatten()))
    # A_final = (get_matrix_from_7d_pose(new_pose)[:3, :4]).T.reshape((12,1))

    # @jit
    # def incremental_ik(q, A, A_final, step=0.1, atol=1e-4):
    #     while True:
    #         delta_A = (A_final - A)
    #         print(np.max(np.abs(delta_A)))
    #         if np.max(np.abs(delta_A)) <= atol:
    #             break
    #         J_q = J_lamb(q[0,0], q[1,0], q[2,0], q[3,0], q[4,0], q[5,0], q[6,0])
    #         J_q = J_q / np.linalg.norm(J_q)  # normalize Jacobian
            
    #         # multiply by step to interpolate between current and target pose
    #         delta_q = np.linalg.pinv(J_q) @ (delta_A*step)
            
    #         q = q + delta_q
    #         A = A_lamb(q[0,0], q[1,0],q[2,0],q[3,0],q[4,0],q[5,0],q[6,0])
    #     return q, np.max(np.abs(delta_A))
    
    # print('step 4...')
    # q, _ = incremental_ik(q_init, A_init, A_final, atol=1e-4)
    # print(q.flatten())