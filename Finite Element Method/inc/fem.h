#ifndef FEM_H
#define FEM_H
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;
class FiniteElementRod{
    double length;    // Length of the rod
    double material_modulus;    // Young's modulus of the material (Pa)
    double load;    // Applied load at x=0 (N)
    int num_elements_;       // Number of elements
    double element_length_;   // Length of each element
    std::vector<double> nodes_;    //// Nodal coordinates
    std::vector<std::pair<int, int>> elements_;   // Element connectivity

   public:
    // Constructor for initializing problem parameters
    FiniteElementRod(double length, double material_modulus, double load);
    // Function to generate the finite element mesh
    void generateMesh(int num_elements);
    // Function to create the local stiffness matrix and assemble it into Global Stiffness Matrix
    // Parameter 'type' specifies uniform or non-uniform area
    MatrixXd assembleStiffnessMatrix(int type);
    // Function to solve the finite element problem and compute displacements
    VectorXd solve(int type);
    // Function to print the displacements at nodal points
    void printDisplacement(VectorXd& displacement);
};
#endif