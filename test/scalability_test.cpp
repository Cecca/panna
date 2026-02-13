#include <iostream>
#include <highfive/H5Easy.hpp>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/distance.hpp"
#include "panna/data.hpp"
#include "panna/rand.hpp"
#include <chrono>

using namespace panna;

int main () {
    seed_global_rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );
    const size_t conc = 12;
    const uint8_t rotations = 3;
    // const size_t dimension = 50;
    const std::vector<size_t> dimensions = {5, 10, 20, 50};//50, 400, 800, 1600};
    const size_t rep = 500;
    const std::vector<size_t> lenghts = {10000, 100000, 1000000};
    using Point = NormedPoints; // UnitNormPoints or NormedPoints
    using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
    using Hasher = E2LSH<conc, Point, Distance>; //LatticeLSH<4, Point, Distance>;
    // using Hasher = CrossPolytope<conc, Point, Distance, rotations>;
    std::ofstream outfile("results/weight_results.csv", std::ios_base::app);

    for (const auto& n : lenghts){
        for (const auto& dimension: dimensions) {
        // Use LatticeLSH (K=4) as the hasher; no explicit builder needed here

        std::vector<std::vector<float>> points;
        for ( size_t i = 0; i < n; i++ ) {
            std::vector<float> point = sample_random_normal_vector( dimension );
            points.push_back( point );
        }

        // Exact computation
        EMST<Point, Hasher, Distance> tree( dimension, rep, points, 0 );
        EMST<Point, Hasher, Distance> approx_tree( dimension, rep, points, 0.1 );
        // auto start = std::chrono::high_resolution_clock::now();
        // float weight = tree.exact_tree();
        // auto end = std::chrono::high_resolution_clock::now();
        // outfile << "Exact, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;

        for ( size_t repet= 0; repet <1 ; repet++) {
            std::chrono::high_resolution_clock::time_point start, end;
            float weight;
            // // Exact with Kruskal+
            start = std::chrono::high_resolution_clock::now();
            std::tie(weight, std::ignore) = tree.find_tree();
            end = std::chrono::high_resolution_clock::now();
            outfile << "K+, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;

            // Approximate with Kruskal+-
            start = std::chrono::high_resolution_clock::now();
            std::tie( weight, std::ignore ) = tree.find_tree();
            end = std::chrono::high_resolution_clock::now();
            outfile << "K+ ɛ 0.1, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;
        }
    }
    }

    outfile.close();


    return 0;
}
