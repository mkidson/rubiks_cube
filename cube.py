import numpy as np
import matplotlib.pyplot as plt
import matplotlib



class cube(object):

    white_face =  np.full((9), 0)
    yellow_face =  np.full((9), 1)
    red_face =  np.full((9), 2)
    orange_face =  np.full((9), 3)
    blue_face =  np.full((9), 4)
    green_face =  np.full((9), 5)
    colours = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

    def __init__(self):

        # 1x54 arrays saving state of the cube
        self.solved_state = np.concatenate((self.green_face, self.blue_face, self.white_face, self.yellow_face, self.red_face, self.orange_face))
        self.curr_state = np.copy(self.solved_state)
        self.move_set = []

#region matrix definitions
        # now the mess of matrix definitions
        self.face_rotate = np.diagflat(np.ones(54))

        self.face_rotate[0,2] = 1
        self.face_rotate[2,8] = 1
        self.face_rotate[8,6] = 1
        self.face_rotate[6,0] = 1

        self.face_rotate[1,5] = 1
        self.face_rotate[5,7] = 1
        self.face_rotate[7,3] = 1
        self.face_rotate[3,1] = 1

        for i in [0,1,2,3,5,6,7,8]:
            self.face_rotate[i,i] = 0

        self.F_turn = np.copy(self.face_rotate)

        for i in [24,25,26,36,39,42,29,28,27,53,50,47]:
            self.F_turn[i,i] = 0

        self.F_turn[24,36] = 1
        self.F_turn[36,29] = 1
        self.F_turn[29,53] = 1
        self.F_turn[53,24] = 1

        self.F_turn[25,39] = 1
        self.F_turn[39,28] = 1
        self.F_turn[28,50] = 1
        self.F_turn[50,25] = 1

        self.F_turn[26,42] = 1
        self.F_turn[42,27] = 1
        self.F_turn[27,47] = 1
        self.F_turn[47,26] = 1


        self.B_turn = np.roll(np.roll(np.copy(self.face_rotate), 9, 0), 9, 1)

        for i in [20,19,18,45,48,51,33,34,35,44,41,38]:
            self.B_turn[i,i] = 0

        self.B_turn[20,45] = 1
        self.B_turn[45,33] = 1
        self.B_turn[33,44] = 1
        self.B_turn[44,20] = 1

        self.B_turn[19,48] = 1
        self.B_turn[48,34] = 1
        self.B_turn[34,41] = 1
        self.B_turn[41,19] = 1

        self.B_turn[18,51] = 1
        self.B_turn[51,35] = 1
        self.B_turn[35,38] = 1
        self.B_turn[38,18] = 1


        self.U_turn = np.roll(np.roll(np.copy(self.face_rotate), 18, 0), 18, 1)

        for i in [11,38,2,47,10,37,1,46,9,36,0,45]:
            self.U_turn[i,i] = 0

        self.U_turn[11,38] = 1
        self.U_turn[38,2] = 1
        self.U_turn[2,47] = 1
        self.U_turn[47,11] = 1

        self.U_turn[10,37] = 1
        self.U_turn[37,1] = 1
        self.U_turn[1,46] = 1
        self.U_turn[46,10] = 1

        self.U_turn[9,36] = 1
        self.U_turn[36,0] = 1
        self.U_turn[0,45] = 1
        self.U_turn[45,9] = 1


        self.D_turn = np.roll(np.roll(np.copy(self.face_rotate), 27, 0), 27, 1)

        for i in [6,42,15,51,7,43,16,52,8,44,17,53]:
            self.D_turn[i,i] = 0

        self.D_turn[6,42] = 1
        self.D_turn[42,15] = 1
        self.D_turn[15,51] = 1
        self.D_turn[51,6] = 1

        self.D_turn[7,43] = 1
        self.D_turn[43,16] = 1
        self.D_turn[16,52] = 1
        self.D_turn[52,7] = 1

        self.D_turn[8,44] = 1
        self.D_turn[44,17] = 1
        self.D_turn[17,53] = 1
        self.D_turn[53,8] = 1


        self.R_turn = np.roll(np.roll(np.copy(self.face_rotate), 36, 0), 36, 1)

        for i in [26,9,35,8,23,12,32,5,20,15,29,2]:
            self.R_turn[i,i] = 0

        self.R_turn[26,9] = 1
        self.R_turn[9,35] = 1
        self.R_turn[35,8] = 1
        self.R_turn[8,26] = 1

        self.R_turn[23,12] = 1
        self.R_turn[12,32] = 1
        self.R_turn[32,5] = 1
        self.R_turn[5,23] = 1

        self.R_turn[20,15] = 1
        self.R_turn[15,29] = 1
        self.R_turn[29,2] = 1
        self.R_turn[2,20] = 1


        self.L_turn = np.roll(np.roll(np.copy(self.face_rotate), 45, 0), 45, 1)

        for i in [18,0,27,17,21,3,30,14,24,6,33,11]:
            self.L_turn[i,i] = 0

        self.L_turn[18,0] = 1
        self.L_turn[0,27] = 1
        self.L_turn[27,17] = 1
        self.L_turn[17,18] = 1

        self.L_turn[21,3] = 1
        self.L_turn[3,30] = 1
        self.L_turn[30,14] = 1
        self.L_turn[14,21] = 1

        self.L_turn[24,6] = 1
        self.L_turn[6,33] = 1
        self.L_turn[33,11] = 1
        self.L_turn[11,24] = 1


        self.M_turn = np.diagflat(np.ones(54))

        for i in [1,28,16,19,4,31,13,22,7,34,10,25]:
            self.M_turn[i,i] = 0

        self.M_turn[1,28] = 1
        self.M_turn[28,16] = 1
        self.M_turn[16,19] = 1
        self.M_turn[19,1] = 1

        self.M_turn[4,31] = 1
        self.M_turn[31,13] = 1
        self.M_turn[13,22] = 1
        self.M_turn[22,4] = 1

        self.M_turn[7,34] = 1
        self.M_turn[34,10] = 1
        self.M_turn[10,25] = 1
        self.M_turn[25,7] = 1


        self.E_turn = np.diagflat(np.ones(54))

        for i in [3,39,12,48,4,40,13,49,5,41,14,50]:
            self.E_turn[i,i] = 0

        self.E_turn[3,39] = 1
        self.E_turn[39,12] = 1
        self.E_turn[12,48] = 1
        self.E_turn[48,3] = 1

        self.E_turn[4,40] = 1
        self.E_turn[40,13] = 1
        self.E_turn[13,49] = 1
        self.E_turn[49,4] = 1

        self.E_turn[5,41] = 1
        self.E_turn[41,14] = 1
        self.E_turn[14,50] = 1
        self.E_turn[50,5] = 1


        self.S_turn = np.diagflat(np.ones(54))

        for i in [21,37,32,52,22,40,31,49,23,43,30,46]:
            self.S_turn[i,i] = 0

        self.S_turn[21,37] = 1
        self.S_turn[37,32] = 1
        self.S_turn[32,52] = 1
        self.S_turn[52,21] = 1

        self.S_turn[22,40] = 1
        self.S_turn[40,31] = 1
        self.S_turn[31,49] = 1
        self.S_turn[49,22] = 1

        self.S_turn[23,43] = 1
        self.S_turn[43,30] = 1
        self.S_turn[30,46] = 1
        self.S_turn[46,23] = 1
#endregion

    def reset_state(self):
        self.curr_state = np.copy(self.solved_state)


#region turn function definitions
    def F(self):
        
        self.curr_state = np.matmul(self.curr_state, self.F_turn)

    def F_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.F_turn.T)
    
    def B(self):
        
        self.curr_state = np.matmul(self.curr_state, self.B_turn)

    def B_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.B_turn.T)

    def U(self):
        
        self.curr_state = np.matmul(self.curr_state, self.U_turn)

    def U_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.U_turn.T)
    
    def D(self):
        
        self.curr_state = np.matmul(self.curr_state, self.D_turn)

    def D_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.D_turn.T)
    
    def R(self):
        
        self.curr_state = np.matmul(self.curr_state, self.R_turn)

    def R_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.R_turn.T)
    
    def L(self):
        
        self.curr_state = np.matmul(self.curr_state, self.L_turn)

    def L_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.L_turn.T)
    
    def M(self):
        
        self.curr_state = np.matmul(self.curr_state, self.M_turn)

    def M_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.M_turn.T)
    
    def E(self):
        
        self.curr_state = np.matmul(self.curr_state, self.E_turn)

    def E_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.E_turn.T)
    
    def S(self):
        
        self.curr_state = np.matmul(self.curr_state, self.S_turn)

    def S_prime(self):

        self.curr_state = np.matmul(self.curr_state, self.S_turn.T)
    
    def X(self):
        self.curr_state = self.L_prime(self.M_prime(self.R()))
    
    def X_prime(self):
        self.curr_state = self.L(self.M(self.R_prime()))

    def Y(self):
        self.curr_state = self.D_prime(self.E_prime(self.U()))

    def Y_prime(self):
        self.curr_state = self.D(self.E(self.U_prime()))

    def Z(self):
        self.curr_state = self.B_prime(self.S(self.F()))

    def Z_prime(self):
        self.curr_state = self.B(self.S_prime(self.F_prime()))

#endregion


    def calculate_metric(self):
        # something about checking for all the entries in the state array, for all places one up/down OR one left/right, how many are the same colour as it. divide that by the number of pieces it's adjacent to. then average that over all pieces. that's your "metric" and all with the same metric are in a macrostate

        testers = [[1,3,4], [0,2,3,4,5], [1,4,5], [0,1,4,6,7], [0,1,2,3,5,6,7,8], [1,2,4,7,8], [3,4,7], [3,4,5,6,8], [4,5,7]]

        test_state = np.copy(self.__create_vis_cube_state())

        total_sum = 0

        for face in test_state:
            face_flat = face.flatten()

            face_sum = 0
            for i in range(9):
                segment_colour = face_flat[i]

                unique, counts = np.unique(face_flat[testers[i]], return_counts=True)
                unique_dict = dict(zip(unique, counts))
                
                try:
                    segment_count = unique_dict[segment_colour]
                except KeyError:
                    segment_count = 0
                segment_frac = segment_count / len(testers[i])
                
                face_sum += segment_frac
            
            face_avg = face_sum / 9

            total_sum += face_avg

        total_avg = total_sum / 6

        return total_avg
    

    def __create_vis_cube_state(self):

        cube_state_split_1 = np.split(np.copy(self.curr_state), 6)
        cube_state_split_final = []

        for i in cube_state_split_1:
            cube_state_split_final.append(np.split(i, 3))

        return np.array(cube_state_split_final)
    

    def __visualise_face(self, face, translation, ax):

        ax.add_patch(plt.Rectangle((translation[0]-1, translation[1]+1), 0.8, 0.8, color=self.colours[int(face[0][0])]))
        ax.add_patch(plt.Rectangle((translation[0]+0, translation[1]+1), 0.8, 0.8, color=self.colours[int(face[0][1])]))
        ax.add_patch(plt.Rectangle((translation[0]+1, translation[1]+1), 0.8, 0.8, color=self.colours[int(face[0][2])]))

        ax.add_patch(plt.Rectangle((translation[0]-1, translation[1]+0), 0.8, 0.8, color=self.colours[int(face[1][0])]))
        ax.add_patch(plt.Rectangle((translation[0]+0, translation[1]+0), 0.8, 0.8, color=self.colours[int(face[1][1])]))
        ax.add_patch(plt.Rectangle((translation[0]+1, translation[1]+0), 0.8, 0.8, color=self.colours[int(face[1][2])]))

        ax.add_patch(plt.Rectangle((translation[0]-1, translation[1]-1), 0.8, 0.8, color=self.colours[int(face[2][0])]))
        ax.add_patch(plt.Rectangle((translation[0]+0, translation[1]-1), 0.8, 0.8, color=self.colours[int(face[2][1])]))
        ax.add_patch(plt.Rectangle((translation[0]+1, translation[1]-1), 0.8, 0.8, color=self.colours[int(face[2][2])]))


    def visualise_state(self, ax):

        ax.add_patch(plt.Rectangle((-15, -15), 30, 30, color='gray'))

        vis_state = self.__create_vis_cube_state()

        # ax.add_patch(plt.Rectangle((0, 0), 0.8, 0.8, color=colours[state[0][0][0]]))
        self.__visualise_face(vis_state[0], [0, 0], ax)
        self.__visualise_face(vis_state[1], [8, 0], ax)
        self.__visualise_face(vis_state[2], [0, 4], ax)
        self.__visualise_face(vis_state[3], [0, -4], ax)
        self.__visualise_face(vis_state[4], [4, 0], ax)
        self.__visualise_face(vis_state[5], [-4, 0], ax)

        metric = self.calculate_metric()

        plt.xlim(-7.5, 12.5)
        plt.ylim(-10, 10)
        plt.title(f'$M_m$ = {metric:.5f}')
        plt.axis('off')
        plt.gca().set_aspect('equal')


    def __button_press(self, event, ax):

        if event.key == "f":
            self.F()
            self.move_set.append('F')

        elif event.key == "F":
            self.F_prime()
            self.move_set.append('F\'')

        elif event.key == "b":
            self.B()
            self.move_set.append('B')

        elif event.key == "B":
            self.B_prime()
            self.move_set.append('B\'')

        elif event.key == "u":
            self.U()
            self.move_set.append('U')
            
        elif event.key == "U":
            self.U_prime()
            self.move_set.append('U\'')

        elif event.key == "d":
            self.D()
            self.move_set.append('D')
            
        elif event.key == "D":
            self.D_prime()
            self.move_set.append('D\'')
        
        elif event.key == "r":
            self.R()
            self.move_set.append('R')
            
        elif event.key == "R":
            self.R_prime()
            self.move_set.append('R\'')

        elif event.key == "l":
            self.L()
            self.move_set.append('L')
            
        elif event.key == "L":
            self.L_prime()
            self.move_set.append('L\'')


        elif event.key == "m":
            self.M()
            self.move_set.append('M')

        elif event.key == "M":
            self.M_prime()
            self.move_set.append('M\'')

        elif event.key == "e":
            self.E()
            self.move_set.append('E')

        elif event.key == "E":
            self.E_prime()
            self.move_set.append('E\'')

        elif event.key == "h":
            self.S()
            self.move_set.append('S')

        elif event.key == "H":
            self.S_prime()
            self.move_set.append('S\'')


        elif event.key == "x":
            self.X()
            self.move_set.append('X')

        elif event.key == "X":
            self.X_prime()
            self.move_set.append('X\'')
            
        elif event.key == "y":
            self.Y()
            self.move_set.append('Y')

        elif event.key == "Y":
            self.Y_prime()
            self.move_set.append('Y\'')

        elif event.key == "z":
            self.Z()
            self.move_set.append('Z')

        elif event.key == "Z":
            self.Z_prime()
            self.move_set.append('Z\'')
        else:
            pass

        ax.cla()

        self.visualise_state(ax)

        return
    

    def button_operation(self):

        fig = plt.figure()

        ax = fig.add_subplot()

        self.visualise_state(ax)


        cip = fig.canvas.mpl_connect('key_press_event', lambda event: self.__button_press(event, ax))

    def get_move_set(self):
        return self.move_set