#include<iostream>
#include<vector>
#include"../inc/fem.h"
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
using namespace Eigen;

int main(int argc, char* argv[]){
        // Define the problem parameters
    double length = 0.5; // Length of the rod
    double material_modulus = 70e9; // Young's modulus of the material (Pa)
    double load = 5000; // Constant load at x=0 (N)
    // Create a FiniteElementRod object
    FiniteElementRod rod(length, material_modulus, load);
    int type=std::stoi(argv[1]);  // to specify uniform or non-uniform cross-section area in the rod
    int num_elements=std::stoi(argv[2]);
    
    auto start = std::chrono::high_resolution_clock::now();

    rod.generateMesh(num_elements);
    VectorXd displacement = rod.solve(type); // generating diaplacements
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Number of elements: " << num_elements << std::endl;
    rod.printDisplacement(displacement);
    double f;
    if(type==1)
	f=12.5e-4*material_modulus*displacement[num_elements-1]/(length/num_elements);
    else
	f =(12.5e-4*(1 + (static_cast<double>(num_elements - 1) / num_elements)) * material_modulus * displacement[num_elements - 1])/(length / num_elements);
	//f=12.5e-4(1+((num_elements-1)/num_elements))*material_modulus*displacement[num_elements-1]/(length/num_elements);

std::cout<<"The reaction force at fixed end= -"<<f<<"N"<<"\n";
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}