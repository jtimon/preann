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
        cout << "[✓] BASIC RULES IMPLEMENTED (low-hanging fruit!)" << endl;
        cout << "    - Bounds checking" << endl;
        cout << "    - Empty square validation" << endl;
        cout << "    - Stone placement" << endl;
        cout << "[✗] Advanced Go rules still needed" << endl << endl;

        cout << "========================================" << endl;
        cout << "WHAT'S NEXT:" << endl;
        cout << "========================================" << endl;
        cout << "Already implemented (from Reversi patterns):" << endl;
        cout << "✓ legalMove() - bounds, player validation, empty square check" << endl;
        cout << "✓ makeMove() - bounds, player validation, stone placement" << endl << endl;
        cout << "Still needed for full Go:" << endl;
        cout << "1. makeMove() - Capture detection and stone removal" << endl;
        cout << "2. legalMove() - Suicide and ko rule checking" << endl;
        cout << "3. GoBoard::computerEstimation() - Simple heuristic for opponent" << endl;
        cout << "4. Board::endGame() override - Detect when game is over (2 passes)" << endl;
        cout << "5. Board::countPoints() override - Score territory + captured stones" << endl << endl;

        cout << "These can be implemented by:" << endl;
        cout << "- Integrating Fuego C++ library (recommended)" << endl;
        cout << "- Custom reimplementation (simpler for basic Go rules)" << endl << endl;

        cout << "NOTE: If we try to run task.test(goAI), it will hit an assertion" << endl;
        cout << "because the game rules are not yet implemented." << endl << endl;

        delete goAI;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << "Exit success." << endl;
    return 0;
}
