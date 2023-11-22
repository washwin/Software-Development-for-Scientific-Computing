#include"../inc/fem.h"
#include<cstdio>
#include <vector>
#include<iostream>
#include <Eigen/Dense>
using namespace Eigen;

//constructor
FiniteElementRod::FiniteElementRod(double length, double material_modulus, double load)
    : length(length), material_modulus(material_modulus), load(load) {
}


void FiniteElementRod::generateMesh(int num_elements) {
    num_elements_ = num_elements;  // Set the number of elements in the rod
    element_length_ = length / num_elements;   // Calculate the length of each element
    nodes_.resize(num_elements_ + 1);   // Resize the vector to store nodal coordinates
    elements_.resize(num_elements_);   // Resize the vector to store element connectivity
    // Generate nodal coordinates for each node
    for (int i = 0; i <= num_elements_; ++i) {
        nodes_[i] = i * element_length_;
    }
    // Define element connectivity by specifying the pairs of nodes that form each element
    for (int i = 0; i < num_elements_; ++i) {
        elements_[i] = std::make_pair(i, i + 1);
    }
}

//generating global stiffness matrix and global force vector
//eleminating last row and column from K because u(l)=0
//u=F*K^(-1)
VectorXd FiniteElementRod::solve(int type) {
    generateMesh(num_elements_);
    MatrixXd K = assembleStiffnessMatrix(type);
    VectorXd F(num_elements_);
    F.setZero();
    F(0) = load;
    // std::cout<<K<<"\n";
    MatrixXd K_mod=K.topLeftCorner(K.rows()-1,K.cols()-1);
    MatrixXd K_mod_inv=K_mod.inverse();
    VectorXd displacement=K_mod_inv*F;
    return displacement;
}

//printing the displacements
void FiniteElementRod::printDisplacement(VectorXd& displacement) {
    std::cout << "Displacement at nodal points:" << std::endl;
    for (int i = 0; i < num_elements_; ++i) {
	std::cout << "Node " << i << ": " << displacement[i]*-1 << " m" << std::endl;
    }
     std::cout<< "Node " <<num_elements_<<": "<<0<<" m"<<std::endl;
}

//generating the global stiffness matrix
MatrixXd FiniteElementRod::assembleStiffnessMatrix(int type) {
	MatrixXd K(num_elements_ + 1, num_elements_ + 1);
	K.setZero();
	//uniform cross-section area
	if(type==1){
	   double A =12.5e-4; 
           for (int i = 0; i < num_elements_; ++i) {
             int x1 = elements_[i].first;
             int x2 = elements_[i].second;
             double L = nodes_[x2] - nodes_[x1];
	     Matrix2d ke;//local stiffness matrix
             ke << 1, -1, -1, 1;
             ke *= A*material_modulus / L;
             K.block(x1, x1, 2, 2) += ke;  //combining local stiffness matrices
          }
	}
	//non-uniform cross-section area
	else{
	    for (int i = 0; i < num_elements_; ++i) {
             int x1 = elements_[i].first;
             int x2 = elements_[i].second;
             double L = nodes_[x2] - nodes_[x1];
	     double A0=12.5e-4;
	     double A=A0*(1+nodes_[x1]/length);
             Matrix2d ke;
             ke << 1, -1, -1, 1;
             ke *= A*material_modulus / L;
             K.block(x1, x1, 2, 2) += ke;
           }
	}
        return K;
}