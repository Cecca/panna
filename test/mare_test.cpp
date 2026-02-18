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
#include <string>

using namespace panna;

int main()  {
    std::string datasets[3] = {
        // "datasets/fashion-mnist-784-euclidean.hdf5",
        // "datasets/glove-100-angular.hdf5",
        // "datasets/nytimes-256-angular.hdf5",
        "datasets/gist-960-euclidean.hdf5",
        "datasets/sift-128-euclidean.hdf5",
        "datasets/deep-image-96-angular.hdf5"
    };
    int index = 0;
    for (const auto& name: datasets) {
        std::cout << "Processing dataset: " << name << std::endl;

        // Parameters
        const size_t conc = 12;
        const uint8_t rotations = 3;
        const size_t dimensions[3] = {960, 128, 96 }; // 784, 100, 256
        float eps[6] = { 0.2, 0.5, 1.0, 5.0, 10.0, 20.0 };
        float probs[4] = {0.01, 0.05, 0.1, 0.2};
        std::vector<float> weigths; 
        using Dataset = NormedPoints; // UnitNormPoints or NormedPoints
        using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
        //using Hasher = LatticeLSH<4, Dataset, Distance>;
        using Hasher = E2LSH<conc, Dataset, Distance>;
        // using Hasher = CrossPolytope<conc, Dataset, Distance, rotations>;
        std::ofstream outfile("weight_results.csv", std::ios_base::app);

        for (const auto& prob: probs) {
            for (const auto& ep: eps) {
                seed_global_rng( 360 );
                H5Easy::File file(name, H5Easy::File::ReadOnly);
                std::vector<std::vector<float>> points =
                    H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
                if (points.size() > 10000)
                    points.resize(10000); // Limit to 1000 points for testing

                    EMST<Dataset, Hasher, Distance> tree(dimensions[index], 500, points, prob, 0);
                    EMST<Dataset, Hasher, Distance> tree_approx(dimensions[index], 500, points, prob, ep);
                    float weight;
                    double duration;

                    for (size_t i = 0; i < 3; i++) {
                        auto time = std::chrono::high_resolution_clock::now();
                        std::tie(weight, std::ignore) = tree.find_tree();
                        duration = std::chrono::duration<double>( std::chrono::high_resolution_clock::now() - time ).count();
                        outfile <<"K+" << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                        


                        time = std::chrono::high_resolution_clock::now();
                        std::tie(weight, std::ignore )= tree_approx.find_tree();
                        duration = std::chrono::duration<double>( std::chrono::high_resolution_clock::now() - time ).count();
                        outfile <<"K± e" << ep << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                    }
                    // Push the writes to file
                    outfile.flush();
                
            }
        }
        index++;
        outfile.close();
    }
}
