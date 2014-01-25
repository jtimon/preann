/*
 * reversiBoard.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiBoard.h"

ReversiBoard::ReversiBoard(ReversiBoard* other) :
        Board(other)
{
}

ReversiBoard::ReversiBoard(unsigned size, BufferType bufferType) :
        Board(size, bufferType)
{
    std::string error;
    if (size % 2 != 0) {
        error = "The size of a Reversi board must be even.";
        throw error;
    }
    if (size < 4) {
        error = "The minimum size of a Reversi board is 4.";
        throw error;
    }
}

ReversiBoard::~ReversiBoard()
{
}

void ReversiBoard::initBoard()
{
    for (int x = 0; x < tSize; ++x) {
        for (int y = 0; y < tSize; ++y) {
            tBoard[x][y] = EMPTY;
        }
    }
    int halfSize = tSize / 2;
    setSquare(halfSize, halfSize, PLAYER_1);
    setSquare(halfSize - 1, halfSize - 1, PLAYER_1);
    setSquare(halfSize - 1, halfSize, PLAYER_2);
    setSquare(halfSize, halfSize - 1, PLAYER_2);
}

bool ReversiBoard::legalMove(unsigned xPos, unsigned yPos, SquareState player)
{
    if (xPos >= tSize || yPos >= tSize) {
        std::string error = "ReversiBoard::legalMove : The position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }
    if (player == EMPTY) {
        std::string error = "ReversiBoard::legalMove : Empty square is not a player.";
        throw error;
    }
    if (getSquare(xPos, yPos) != EMPTY) {
        return false;
    }
    unsigned x = 0, y = 0;

    for (int a = -1; a <= 1; a++) { //for each direction (left, right)
        for (int b = -1; b <= 1; b++) { // for each direction (up down)
            if (a == 0 && b == 0) {
                break;
            } else {
                x = xPos + a;
                y = yPos + b;
            }
            //at least one of the squares has to belong to the opponent
            while (squareIs(x, y, opponent(player))) {

                x += a;
                y += b;
                if (squareIs(x, y, player)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void ReversiBoard::makeMove(unsigned xPos, unsigned yPos, SquareState player)
{
    if (xPos >= tSize || yPos >= tSize) {
        std::string error = "ReversiBoard::makeMove : The position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }
    if (player == EMPTY) {
        std::string error = "ReversiBoard::makeMove : Empty square is not a player.";
        throw error;
    }
    if (getSquare(xPos, yPos) != EMPTY) {
        std::string error = "ReversiBoard::makeMove : the square at position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is already occupied";
        throw error;
    }
    unsigned x = 0, y = 0;

    for (int a = -1; a <= 1; a++) { //for each direction (left, right)
        for (int b = -1; b <= 1; b++) { // for each direction (up down)
            if (a == 0 && b == 0) {
                break;
            } else {
                x = xPos + a;
                y = yPos + b;
            }
            //at least one of the squares has to belong to the opponent
            while (squareIs(x, y, opponent(player))) {

                x += a;
                y += b;
                if (squareIs(x, y, player)) {

                    x -= a;
                    y -= b;
                    // set the squares inbetween to the player
                    while (x != xPos || y != yPos) {
                        tBoard[x][y] = player;
                        x -= a;
                        y -= b;
                    }
                    tBoard[x][y] = player;
                    break;
                }
            }
        }
    }
}

float ReversiBoard::individualEstimation(unsigned xPos, unsigned yPos, SquareState player,
                                         Individual* individual)
{
    ReversiBoard* futureBoard = new ReversiBoard(this);
    futureBoard->makeMove(xPos, yPos, player);
    individual->updateInput(0, futureBoard->updateInterface());
    individual->calculateOutput();
    delete (futureBoard);

    // the first element of the last layer
    return individual->getOutput(individual->getNumLayers() - 1)->getElement(0);
}

float ReversiBoard::computerEstimation(unsigned xPos, unsigned yPos, SquareState player)
{
    int x, y;
    float quality = 0;

    for (int a = -1; a <= 1; a++) { //for each direction (left, right)
        for (int b = -1; b <= 1; b++) { // for each direction (up down)
            if (a == 0 && b == 0) {
                break;
            } else {
                x = xPos + a;
                y = yPos + b;
            }

            //at least one of the squares has to belong to the opponent
            while (squareIs(x, y, opponent(player))) {

                x += a;
                y += b;
                if (squareIs(x, y, player)) {
                    x -= a;
                    y -= b;
                    // count quality
                    while (x != xPos && y != yPos) {
                        ++quality;
                        x -= a;
                        y -= b;
                    }
                    break;
                }
            }
        }
    }
    return quality;
}
