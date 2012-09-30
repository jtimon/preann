/*
 * board.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "board.h"

void Board::baseConstructor(unsigned size)
{
    tSize = size;
    tBoard = (SquareState**) MemoryManagement::malloc(sizeof(SquareState*) * tSize);
    for (unsigned i = 0; i < tSize; ++i) {
        tBoard[i] = (SquareState*) MemoryManagement::malloc(sizeof(SquareState) * tSize);
    }

    tInterface = new Interface(tSize * tSize * 2, BT_BIT);
    initBoard();
}

Board::Board(unsigned size)
{
    this->baseConstructor(size);
}

Board::Board(Board* other)
{
    this->baseConstructor(other->getSize());

    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {
            tBoard[x][y] = other->getSquare(x, y);
        }
    }
}

Board::~Board()
{
    for (unsigned i = 0; i < tSize; ++i) {
        MemoryManagement::free(tBoard[i]);
    }
    MemoryManagement::free(tBoard);
    delete (tInterface);
}

void Board::initBoard()
{
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {
            tBoard[x][y] = EMPTY;
        }
    }
}

bool Board::insideBoard(int x, int y)
{
    return x >= 0 && x < tSize && y >= 0 && y < tSize;
}

bool Board::squareIs(int x, int y, SquareState squareState)
{
    if (!insideBoard(x, y)) {
        return false;
    }
    return tBoard[x][y] == squareState;
}

unsigned Board::getSize()
{
    return tSize;
}

void Board::setSquare(unsigned x, unsigned y, SquareState squareState)
{
    if (!insideBoard(x, y)) {
        std::string error = "Board::setSquare : The position (" + to_string(x) + ", " + to_string(y)
                + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }
    tBoard[x][y] = squareState;
}

SquareState Board::getSquare(unsigned x, unsigned y)
{
    if (!insideBoard(x, y)) {
        std::string error = "Board::getSquare : The position (" + to_string(x) + ", " + to_string(y)
                + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }
    return tBoard[x][y];
}

Interface* Board::getInterface()
{
    return tInterface;
}

Interface* Board::updateInterface()
{
    unsigned index = 0;
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {

            float play1 = 0, play2 = 0;
            if (tBoard[x][y] == PLAYER_1) {
                play1 = 1;
            } else if (tBoard[x][y] == PLAYER_2) {
                play2 = 1;
            }
            tInterface->setElement(index++, play1);
            tInterface->setElement(index++, play2);
        }
    }
    return tInterface;
}

bool Board::canMove(SquareState player)
{
    if (player == EMPTY) {
        std::string error = "ReversiBoard::canMove : Empty square is not a player.";
        throw error;
    }
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {

            if (legalMove(x, y, player)) {
                return true;
            }
        }
    }
    return false;
}

void Board::turn(SquareState player, Individual* individual)
{
    if (player == EMPTY) {
        std::string error = "ReversiBoard::turn : Empty square is not a player.";
        throw error;
    }
    float maxQuality = 0;
    vector<Move> moves;
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {

            if (legalMove(x, y, player)) {
                Move move;
                move.xPos = x;
                move.yPos = y;
                if (individual == NULL) {
                    move.quality = computerEstimation(x, y, player);
                } else {
                    move.quality = individualEstimation(x, y, player, individual);
                }
                if (move.quality >= maxQuality || moves.size() == 0) {
                    maxQuality = move.quality;
                    moves.push_back(move);
                }
            }
        }
    }
    vector<Move> bestMoves;
    for (int i = 0; i < moves.size(); ++i) {
        if (moves[i].quality == maxQuality) {
            bestMoves.push_back(moves[i]);
        }
    }
    if (bestMoves.size() > 0) {
        Move chosenMove = bestMoves[Random::positiveInteger(bestMoves.size())];
        makeMove(chosenMove.xPos, chosenMove.yPos, player);
    }
}

bool Board::endGame()
{
    if (canMove(PLAYER_1) || canMove(PLAYER_2)) {
        return false;
    }
    return true;
}

int Board::countPoints(SquareState player)
{
    int points = 0;
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {
            if (tBoard[x][y] == player) {
                ++points;
            } else if (tBoard[x][y] == opponent(player)) {
                --points;
            }
        }
    }
    return points;
}


void Board::print()
{
    cout<<"---------------------------------------"<<endl;
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {
            if (tBoard[x][y] == PLAYER_1) {
                cout<<" X";
            } else if (tBoard[x][y] == PLAYER_2) {
                cout<<" O";
            } else {
                cout<<" .";
            }
        }
        cout<<endl;
    }
    cout<<"---------------------------------------"<<endl;
}

SquareState Board::opponent(SquareState player)
{
    if (player == EMPTY) {
        std::string error = "Board::adversary : Empty square is not a player.";
        throw error;
    }
    else if (player == PLAYER_1) {
        return PLAYER_2;
    }
    else {
        return PLAYER_1;
    }
}

