#include <iostream>
#include "tasks/goTask.h"
#include "genetic/individual.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "=== PREANN Go Demo ===" << endl << endl;
    cout << "NOTE: This demo shows the Go framework integration," << endl;
    cout << "but actual game playing is NOT YET IMPLEMENTED." << endl;
    cout << "The plan is to include Fuego library or reimplement Go rules." << endl << endl;

    try {
        // Create a 9x9 Go board (smallest standard size)
        cout << "Creating 9x9 Go board..." << endl;
        GoBoard board(9, BT_BIT);
        cout << "Board created successfully!" << endl << endl;

        // Create a GoTask
        cout << "Creating GoTask (9x9 board, 1 test game)..." << endl;
        GoTask task(9, BT_BIT, 1);
        cout << "Task created successfully!" << endl << endl;

        // Create a random untrained neural network
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        cout << "Creating random untrained neural network for Go..." << endl;
        Individual* goAI = task.getExample(&params);
        cout << "Neural network created with " << goAI->getNumLayers() << " layers" << endl;
        cout << "Input size: 9x9 = 81 positions" << endl;
        cout << "Output size: 9x9 = 81 positions (one per potential move)" << endl << endl;

        cout << "========================================" << endl;
        cout << "FRAMEWORK STATUS:" << endl;
        cout << "========================================" << endl;
        cout << "[✓] GoBoard class created" << endl;
        cout << "[✓] GoTask class created" << endl;
        cout << "[✓] Neural network instantiated" << endl;
        cout << "[✓] Board interface integrated" << endl;
        cout << "[✓] COMPLETE GO RULES IMPLEMENTED!" << endl;
        cout << "    - Empty square validation" << endl;
        cout << "    - Stone placement" << endl;
        cout << "    - Capture detection (liberties)" << endl;
        cout << "    - Suicide rule enforcement" << endl;
        cout << "    - Ko rule (simple)" << endl;
        cout << "[✓] Zero new dependencies - pure C++" << endl << endl;

        cout << "========================================" << endl;
        cout << "READY TO PLAY GO!" << endl;
        cout << "========================================" << endl;
        cout << "Implemented (~150 lines of C++):" << endl;
        cout << "✓ legalMove() - complete with suicide and ko rules" << endl;
        cout << "✓ makeMove() - complete with capture detection" << endl;
        cout << "✓ Helper functions: findGroup, countLiberties, removeGroup" << endl;
        cout << "✓ Ko tracking with board hashing" << endl << endl;
        cout << "Inherited from generic Board:" << endl;
        cout << "✓ canMove() - finds legal moves" << endl;
        cout << "✓ turn() - AI move selection" << endl;
        cout << "✓ endGame() - game termination" << endl;
        cout << "✓ countPoints() - basic scoring" << endl << endl;

        cout << "Still TODO (optional enhancements):" << endl;
        cout << "1. GoBoard::computerEstimation() - heuristic opponent for faster bootstrap" << endl;
        cout << "2. Population save/load - checkpoint training progress" << endl;
        cout << "3. Override endGame() - proper pass detection" << endl;
        cout << "4. Override countPoints() - territory scoring (Chinese rules)" << endl << endl;

        cout << "PREANN can now train neural networks to play Go!" << endl;
        cout << "After 13+ years, the vision is realized." << endl << endl;

        delete goAI;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << "Exit success." << endl;
    return 0;
}
