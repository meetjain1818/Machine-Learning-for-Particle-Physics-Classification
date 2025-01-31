import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class preprocessData():
    '''
    columns = ['isophoton_Eta', 'isophoton_Phi', 'jet1_Eta', 'jet1_Phi','jet2_Eta', 'jet2_Phi']
    '''
    def __init__(self):
        pass

    def shift_origin(self,x:pd.DataFrame) -> pd.DataFrame:
        #TODO: Take coordinates and subtract origin coordinates from it.
        # 1. Storing isophoton coordinates as origin
        origin = x.loc[:,['isophoton_Eta', 'isophoton_Phi']].values
    
        # 2. Subtracting origin from cooridnates of isophoton, jet1, jet2
        isophoton_shift_coor = x.loc[:,['isophoton_Eta', 'isophoton_Phi']].values - origin
        jet1_shift_coor = x.loc[:,['jet1_Eta', 'jet1_Phi']].values - origin
        jet2_shift_corr = x.loc[:,['jet2_Eta', 'jet2_Phi']].values - origin
    
        # Returning a new dataframe after concatenating updated coordinates
        return pd.DataFrame(np.concatenate([isophoton_shift_coor, jet1_shift_coor, jet2_shift_corr], axis = 1), columns = ['isophoton_Eta', 'isophoton_Phi', 'jet1_Eta', 'jet1_Phi', 'jet2_Eta', 'jet2_Phi'])

    def rotate_coordinates(self,x:pd.DataFrame) -> pd.DataFrame:
        #TODO: Rotate the plane such that Jet1 lies below isophoton(origin)
        y_axis = 0,-1 # -Y axis
        rotated_coordinates = pd.DataFrame(x[['isophoton_Eta', 'isophoton_Phi']])
        jet1_rotated = []
        jet2_rotated = []
        # Loop for rotating each row by the corresponding theta
        for i in range(len(x)):
            row = x.iloc[i]
            # calculating theta using dot product between jet1 vector and -Y axis
            theta = np.arccos(np.dot(row.loc[['jet1_Eta', 'jet1_Phi']].values, y_axis)/np.linalg.norm(row.loc[['jet1_Eta', 'jet1_Phi']]))
            if row.loc['jet1_Eta'] < 0:
                theta = -theta
            # Rotation matrix that rotates clockwise by theta
            rotation_matrix = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
            ])
            # Operating rotation matrix on jet1 and jet2
            jet1_rotated.append((rotation_matrix @ (row.loc[['jet1_Eta', 'jet1_Phi']].values).T).T)
            jet2_rotated.append((rotation_matrix @ (row.loc[['jet2_Eta', 'jet2_Phi']].values).T).T)
            
        jet1_rotated = np.array(jet1_rotated) # list to array
        jet2_rotated = np.array(jet2_rotated)
        # Storing the rotated data as columns of a new dataframe
        rotated_coordinates['jet1_Eta'] = jet1_rotated[:,0]
        rotated_coordinates['jet1_Phi'] = jet1_rotated[:,1]
        rotated_coordinates['jet2_Eta'] = jet2_rotated[:,0]
        rotated_coordinates['jet2_Phi'] = jet2_rotated[:,1]
    
        return rotated_coordinates

    def bound_phi(self, x:pd.DataFrame) -> pd.DataFrame:
        '''
        Normalize angles to the range (-π, π)

        This function takes angular values in any range and normalizes them to lie 
        within (-π, π) using the following steps:
        
        1. Transform negative angles to positive:
           - Add 2π to negative angles to get angles in range (0, 2π)
           
        2. Map angles larger than π to their equivalent in (-π, 0):
           - For angles > π: result = -(2π - angle)
        '''
        jet1_phi = x['jet1_Phi'].values
        jet2_phi = x['jet2_Phi'].values
        isophoton_phi = x['isophoton_Phi'].values

        updated_phi = np.zeros((len(x), 3))
        object_iterator = iter([jet1_phi, jet2_phi, isophoton_phi])
        for col, obj in enumerate(object_iterator):
            for row, phi in enumerate(obj):
                if phi < 0:
                    updated_phi[row, col] = 2 * np.pi + phi
                else:
                    updated_phi[row, col] = phi

        for idx, phi in np.ndenumerate(updated_phi):
            if phi > np.pi:
                updated_phi[idx[0], idx[1]] = -1 * (2 * np.pi - updated_phi[idx[0], idx[1]])
        
        updated_x = x.copy()
        updated_x['jet1_Phi'] = updated_phi[:,0]
        updated_x['jet2_Phi'] = updated_phi[:,1]
        updated_x['isophoton_Phi'] = updated_phi[:,2]

        return updated_x

    def plot_eflow_objects(self, x, ax, *, title: str = 'Plot'):
        """
        Plot eflow objects on a specified subplot
        
        Parameters:
            x: data to plot
            ax: matplotlib axes object to plot on
            title: plot title (optional)
        """
        # Add horizontal and vertical lines
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        # Add horizontal lines at -π and π
        ax.axhline(-np.pi, color='red', linewidth=0.5, linestyle = '--')
        ax.axhline(np.pi, color='red', linewidth=0.5, linestyle = '--')
        
        # Add gridlines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set axis limits
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        
        # Create scatter plots
        ax.scatter(x.loc['isophoton_Eta'], x.loc['isophoton_Phi'], 
                  label=f"Isophoton:({x.loc['isophoton_Eta']:.2f},{x.loc['isophoton_Phi']:.2f})")
        ax.scatter(x.loc['jet1_Eta'], x.loc['jet1_Phi'], 
                  label=f"Jet1:({x.loc['jet1_Eta']:.2f},{x.loc['jet1_Phi']:.2f})", marker = 'x')
        ax.scatter(x.loc['jet2_Eta'], x.loc['jet2_Phi'], 
                  label=f"Jet2:({x.loc['jet2_Eta']:.2f},{x.loc['jet2_Phi']:.2f})", marker = '.')
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel("Eta")
        ax.set_ylabel("Phi")
        ax.legend()
        plt.tight_layout()


    def plot_all(self, x, ax, *, title: str = 'Collective Plot'):
        """
        Plot all data points on a specified subplot
        
        Parameters:
            x: DataFrame containing the data to plot
            ax: matplotlib axes object to plot on
            title: plot title (optional)
        """
        # Add horizontal and vertical lines
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        # Add horizontal lines at -π and π
        ax.axhline(-np.pi, color='red', linewidth=0.5, linestyle = '--')
        ax.axhline(np.pi, color='red', linewidth=0.5, linestyle = '--')
        
        # Add gridlines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set axis limits
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        
        # Create scatter plots
        ax.scatter(x.loc[:,'isophoton_Eta'], x.loc[:,'isophoton_Phi'], label='Isophoton')
        ax.scatter(x.loc[:,'jet1_Eta'], x.loc[:,'jet1_Phi'], label='Jet1', marker='x')
        ax.scatter(x.loc[:,'jet2_Eta'], x.loc[:,'jet2_Phi'], label='Jet2', marker='.', alpha = 0.5)
        
        # Set labels and title
        ax.set_xlabel("Eta")
        ax.set_ylabel("Phi")
        ax.set_title(title)
        ax.legend()


    def Euclidean_distance(self, x:pd.DataFrame , obj1:str = 'jet1', obj2:str = 'jet2') -> np.array:
        '''
        Calculate the Euclidean distance between 
        '''
        obj1_eta_phi = x[[obj1+'_Eta', obj1+'_Phi']].values
        obj2_eta_phi = x[[obj2+'_Eta', obj2+'_Phi']].values

        euclidean_distance = np.sqrt(np.sum((obj1_eta_phi - obj2_eta_phi)**2, axis = 1))
        return euclidean_distance


    def complete_transformation(self, x:pd.DataFrame,*, intermediate_bound_phi = True, final_bound_phi = True) -> pd.DataFrame:
        '''
        Function to do the origin shift, bounding phi, rotation and then bounding phi(optional) in on go.
        '''
        x_shifted = self.shift_origin(x)
        x_bounded = self.bound_phi(x_shifted)
        x_rotated = self.rotate_coordinates(x_bounded)

        if final_bound_phi:
            x_final = self.bound_phi(x_rotated)
            return x_final
        return x_rotated
        