#include <iostream>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <stack>
#include <sys/time.h>
#define COLS 7
#define ROWS 6
#define S 3
#define TABLE_SIZE 100000

unsigned long long getTimeMicrosec()
{
    timeval NOW;
    gettimeofday(&NOW, NULL);
    return NOW.tv_sec * 1000000LL + NOW.tv_usec;
}

struct Position
{
    int moves;
    int board[ROWS][COLS];
    int height[COLS];
    int legalMoves[COLS];
    int numLegalMoves;
    int evaluatedMoves;

    __host__ __device__ bool operator==(const Position &rhs) const
    {
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                if (board[i][j] != rhs.board[i][j])
                {
                    return false;
                }
            }
        }
        return true;
    }
};

struct PositionEntry
{
    Position position;
    int score;
};

template <>
struct std::hash<Position>
{
    size_t operator()(const Position &p) const
    {
        std::string board = "";
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < ROWS; j++)
            {
                board += p.board[i][j];
            }
        }
        return std::hash<std::string>{}(board);
    }
};

__host__ __device__ int hash(Position &p)
{
    int hash = 1;
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            hash += 37 * p.board[i][j] + (i * 31 + j * 29);
        }
    }
    return hash;
}

struct StackEntry
{
    Position position;
    int alpha;
    int beta;
    int depth;
    int value;
    bool up;
};

struct Stack
{
    int top;
    StackEntry entries[43];
};

__device__ void push(Stack *stack, StackEntry entry)
{
    stack->top++;
    stack->entries[stack->top] = entry;
}

__device__ void pop(Stack *stack)
{
    if (stack->top >= 0)
    {
        stack->top--;
    }
    else
    {
        // std::cerr << "Popped empty stack" << std::endl;
    }
}

__device__ StackEntry top(Stack *stack)
{
    return stack->entries[stack->top];
}

__device__ int size(Stack *stack)
{
    return stack->top + 1;
}

__device__ bool empty(Stack *stack)
{
    return size(stack) == 0;
}

__host__ __device__ void initPosition(Position *position)
{
    position->moves = 0;

    position->evaluatedMoves = 0;
    position->numLegalMoves = 0;

    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            position->board[i][j] = 0;
        }
    }

    for (int i = 0; i < COLS; i++)
    {
        position->height[i] = 0;
    }
}

__host__ __device__ void copyPosition(Position *lhs, Position *rhs)
{
    lhs->moves = rhs->moves;

    lhs->evaluatedMoves = rhs->evaluatedMoves;
    lhs->numLegalMoves = rhs->numLegalMoves;

    for (int i = 0; i < lhs->numLegalMoves; i++)
    {
        lhs->legalMoves[i] = rhs->legalMoves[i];
    }

    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            lhs->board[i][j] = rhs->board[i][j];
        }
    }

    for (int i = 0; i < COLS; i++)
    {
        lhs->height[i] = rhs->height[i];
    }
}

__host__ __device__ bool canPlay(Position *position, int col)
{
    return col < COLS && position->height[col] < ROWS;
}

__host__ __device__ void play(Position *position, int col)
{
    position->board[position->height[col]][col] = 1 + position->moves % 2;
    position->height[col]++;
    position->moves++;
}

__host__ void setPosition(Position *p, std::string positionString)
{
    for (int i = 0; i < positionString.size(); i++)
    {
        int pieceColumn = positionString[i] - '1';

        if (pieceColumn > COLS)
        {
            std::cout << "Invalid Position: Invalid column in position: " << pieceColumn << ", at position[" << i << "]. Skipping." << std::endl;
            continue;
        }

        if (!canPlay(p, pieceColumn))
        {
            std::cout << "Invalid Position: Too many pieces in column " << pieceColumn << ", at position[" << i << "]. Skipping." << std::endl;
            continue;
        }

        play(p, pieceColumn);
    }
}

__host__ __device__ void printBoard(Position *p)
{
    for (int i = ROWS - 1; i > -1; i--)
    {
        for (int j = 0; j < COLS; j++)
        {
            printf(" | %d", p->board[i][j]);
        }
        printf("|\n");
    }
}

__host__ __device__ bool isWinningMove(Position *position, int col)
{
    int player = 1 + position->moves % 2;
    // check for vertical alignments
    if (position->height[col] >= 3 && position->board[position->height[col] - 1][col] == player && position->board[position->height[col] - 2][col] == player && position->board[position->height[col] - 3][col] == player)
    {
        return true;
    }

    for (int dy = -1; dy <= 1; dy++)
    {               // Iterate on horizontal (dy = 0) or two diagonal directions (dy = -1 or dy = 1).
        int nb = 0; // counter of the number of stones of current player surronding the played stone in tested direction.
        for (int dx = -1; dx <= 1; dx += 2)
        { // count continuous stones of current player on the left, then right of the played column.
            for (int x = col + dx, y = position->height[col] + dx * dy; x >= 0 && x < COLS && y >= 0 && y < ROWS && position->board[y][x] == player; nb++)
            {
                x += dx;
                y += dx * dy;
            }
        }
        if (nb >= 3)
        {
            return true;
        } // there is an aligment if at least 3 other stones of the current user are surronding the played stone in the tested direction.
    }
    return false;
}

__device__ void generate(Position *position)
{
    int numLegalMoves = 0;
    for (int i = 0; i < COLS; i++)
    {
        if (canPlay(position, i))
        {
            position->legalMoves[numLegalMoves] = i;
            numLegalMoves++;
        }
    }
    position->evaluatedMoves = 0;
    position->numLegalMoves = numLegalMoves;
}

__device__ int getMove(Position *position)
{
    if (position->numLegalMoves == 0 || position->evaluatedMoves >= position->numLegalMoves)
    {
        return -1;
    }
    else
    {
        int move = position->legalMoves[position->evaluatedMoves];
        position->evaluatedMoves++;
        return move;
    }
}

__device__ int negamax(Position position, PositionEntry *hashMap)
{
    // Create stack
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    Stack stack;
    stack.top = -1;

    // Create innitial stack entry
    StackEntry initialEntry;
    initialEntry.position = position;
    initialEntry.alpha = -(COLS * ROWS) / 2;
    initialEntry.beta = (COLS * ROWS) / 2;
    initialEntry.up = false;
    initialEntry.depth = 30;

    // Push
    push(&stack, initialEntry);

    int returnValue = -1000;

    while (size(&stack) >= 0)
    {
        // std::cout << "Stack size: " << stack.size() << std::endl;
        StackEntry topEntry = top(&stack);
        pop(&stack);
        if (topEntry.up)
        {

            // If we're going up, simple, use values in the next entry in stack.
            int score = -topEntry.value;
            topEntry = top(&stack);
            pop(&stack);

            topEntry.up = false;

            // We calculated a score.

            if (score >= topEntry.beta)
            {
                topEntry.value = score;
                PositionEntry entry;
                copyPosition(&entry.position, &topEntry.position);
                entry.score = score;
                int hashIndex = hash(topEntry.position);
                hashMap[hashIndex % TABLE_SIZE] = entry;
                if (size(&stack) == 0)
                {
                    returnValue = score;
                    break;
                }
                topEntry.up = true;
            }

            if (!topEntry.up)
            {
                if (score > topEntry.alpha)
                {
                    // std::cout << "New Alpha: " << score << std::endl;
                    topEntry.alpha = score;
                }
            }

            // Otherwise, we have to generate the next of the position and go down

            int nextMove = getMove(&topEntry.position);
            if (!topEntry.up)
            {
                if (nextMove == -1)
                {
                    topEntry.value = topEntry.alpha;
                    PositionEntry entry;
                    copyPosition(&entry.position, &topEntry.position);
                    entry.score = topEntry.alpha;
                    int hashIndex = hash(topEntry.position);
                    hashMap[hashIndex % TABLE_SIZE] = entry;
                    if (size(&stack) == 0)
                    {
                        returnValue = topEntry.alpha;
                        break;
                    }
                    topEntry.up = true;
                }
            }

            if (topEntry.up)
            {
                StackEntry returnEntry;
                returnEntry.value = topEntry.value;
                returnEntry.up = true;
                push(&stack, returnEntry);
                continue;
            }

            Position nextPosition;
            copyPosition(&nextPosition, &topEntry.position);
            play(&nextPosition, nextMove);

            StackEntry newStackEntry;
            newStackEntry.position = nextPosition;
            newStackEntry.alpha = -topEntry.beta;
            newStackEntry.beta = -topEntry.alpha;
            newStackEntry.depth = topEntry.depth - 1;
            newStackEntry.up = false;

            push(&stack, topEntry);
            push(&stack, newStackEntry);
        }
        else
        {
            // Otherwise, we're going down and we have to keep generating nodes.

            // First, check if we're at a leaf, if we are, we can start going up and reset the top entry

            int hashIndex = hash(topEntry.position);
            PositionEntry entry = hashMap[hashIndex % TABLE_SIZE];
            if (entry.position == topEntry.position)
            {
                printf("Hash Entry HIT!!!!\n");
                topEntry.value = entry.score;
                topEntry.up = true;
            }

            // LEAF COND 1: Draw game.
            if (!topEntry.up)
            {
                if (topEntry.position.moves == ROWS * COLS)
                {
                    // std::cout << "Draw game found." << std::endl;
                    topEntry.value = 0;
                    topEntry.up = true;
                    PositionEntry entry;
                    copyPosition(&entry.position, &topEntry.position);
                    entry.score = 0;
                    int hashIndex = hash(topEntry.position);
                    hashMap[hashIndex % TABLE_SIZE] = entry;
                }
            }

            // LEAF COND 2: Next move has a winning move.
            if (!topEntry.up)
            {
                for (int i = 0; i < COLS; i++)
                {
                    if (canPlay(&topEntry.position, i) && isWinningMove(&topEntry.position, i))
                    {
                        topEntry.value = (COLS * ROWS + 1 - topEntry.position.moves) / 2;
                        topEntry.up = true;
                        PositionEntry entry;
                        copyPosition(&entry.position, &topEntry.position);
                        entry.score = (COLS * ROWS + 1 - topEntry.position.moves) / 2;
                        int hashIndex = hash(topEntry.position);
                        hashMap[hashIndex % TABLE_SIZE] = entry;
                        break;
                    }
                }
            }

            // LEAF COND 3: Beta prune.
            if (!topEntry.up)
            {
                int max = (COLS * ROWS - 1 - topEntry.position.moves) / 2; // upper bound of our score as we cannot win immediately

                // Quick check, if beta is bigger than possible max, then prune.
                if (topEntry.beta > max)
                {
                    topEntry.beta = max; // there is no need to keep beta above our max possible score.
                    // std::cout << "New Beta: " << topEntry.beta << std::endl;
                    if (topEntry.alpha >= topEntry.beta)
                    {
                        // std::cout << "Prune beta exploration" << std::endl;
                        topEntry.value = topEntry.beta; // prune the exploration if the [alpha;beta] window is empty.
                        topEntry.up = true;
                        PositionEntry entry;
                        copyPosition(&entry.position, &topEntry.position);
                        entry.score = max;
                        int hashIndex = hash(topEntry.position);
                        hashMap[hashIndex % TABLE_SIZE] = entry;
                    }
                }
            }

            // LEAF COND 4: We're at our depth limit.
            if (!topEntry.up)
            {
                if (topEntry.depth == 0)
                {
                    topEntry.value = 0;
                    topEntry.up = true;
                }
            }

            if (topEntry.up)
            {
                if (size(&stack) == 0)
                {
                    returnValue = topEntry.value;
                    break;
                }
                StackEntry returnEntry;
                returnEntry.value = topEntry.value;
                returnEntry.up = true;
                push(&stack, returnEntry);
                continue;
            }

            // Otherwise, we're not at the top, so we're going to need to go down and add first move to the stack
            StackEntry newStackEntry;
            generate(&topEntry.position);
            int firstMove = getMove(&topEntry.position);
            Position newStackPosition;
            copyPosition(&newStackPosition, &topEntry.position);
            play(&newStackPosition, firstMove);
            newStackEntry.position = newStackPosition;
            newStackEntry.alpha = -topEntry.beta;
            newStackEntry.beta = -topEntry.alpha;
            newStackEntry.depth = topEntry.depth - 1;
            newStackEntry.up = false;
            push(&stack, topEntry);
            push(&stack, newStackEntry);
        }
    }
    return returnValue;
}

__global__ void parallelGTS(Position *positions, int *scores, int numLeaves, PositionEntry *hashMap)
{
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalIdx < numLeaves && scores[globalIdx] == 22)
    {
        scores[globalIdx] = negamax(positions[globalIdx], hashMap);
    }
}

__host__ int finalNegamax(Position *p, int alpha, int beta, std::unordered_map<Position, int> &leafScores, int depth)
{

    if (p->moves == ROWS * COLS)
    {
        return 0;
    }

    for (int i = 0; i < COLS; i++)
    {
        if (canPlay(p, i) && isWinningMove(p, i))
        {
            return (COLS * ROWS + 1 - p->moves) / 2;
        }
    }

    int max = (COLS * ROWS - 1 - p->moves) / 2;
    if (beta > max)
    {
        beta = max;
        if (alpha >= beta)
        {
            return beta;
        }
    }

    if (depth == S)
    {
        Position lookupPos;
        copyPosition(&lookupPos, p);
        return leafScores[lookupPos];
    }

    for (int i = 0; i < COLS; i++)
    {
        if (canPlay(p, i))
        {
            int score;
            Position newPosition;
            copyPosition(&newPosition, p);
            play(&newPosition, i);
            score = -finalNegamax(&newPosition, -beta, -alpha, leafScores, depth + 1);
            if (score >= beta)
            {
                return score;
            }

            if (score > alpha)
            {
                alpha = score;
            }
        }
    }
    return alpha;
}

int main()
{

    int h_scores[7 * 7 * 7 * 7 * 7 * 7 * 7];
    Position h_positions[7 * 7 * 7 * 7 * 7 * 7 * 7];

    for (int i = 0; i < 7 * 7 * 7 * 7 * 7 * 7 * 7; i++)
    {
        h_scores[i] = 22;
    }

    std::unordered_map<Position, int> scoreTable;

    // Notes: implement BFS, for each leaf node based on depth found, we stop
    // If we reach certain depth, we add node to vector
    // Once done iterating, we copy this over to GPU
    // GPU does rest of parallelization
    // returns back score, each index has score of corresponding node from original vector
    // Iterate through score vector and work vector to populate value hashmap
    // Use as table and redo alpha beta pruning to that depth

    Position initialPosition;
    initPosition(&initialPosition);
    // SET POSITION HERE
    std::string position = "3513265333547163177727167665521";
    setPosition(&initialPosition, position);

    std::cout << "Initial Position: " << std::endl;
    printBoard(&initialPosition);

    int numPositions = 0;
    int numLeaves = 0;
    int score;

    unsigned long long startTime = getTimeMicrosec();
    // BFS
    std::stack<Position> stack = std::stack<Position>();
    stack.push(initialPosition);
    while (!stack.empty())
    {
        Position current = stack.top();
        stack.pop();
        int depth = current.moves - initialPosition.moves;
        // Check if draw.
        if (current.moves == ROWS * COLS)
        {
            if (depth == S)
            {
                scoreTable[current] = 0;
                numLeaves++;
            }
            continue;
        }

        // Check if win.
        bool flag = false;
        for (int i = 0; i < COLS; i++)
        {
            if (canPlay(&current, i) && isWinningMove(&current, i))
            {
                if (depth == S)
                {
                    scoreTable[current] = (COLS * ROWS + 1 - current.moves) / 2;
                    numLeaves++;
                    flag = true;
                    break;
                }
            }
        }
        if (flag)
        {
            continue;
        }

        // Check if depth.
        if (depth == S)
        {
            scoreTable[current] = -23;
            copyPosition(&h_positions[numPositions], &current);
            numLeaves++;
            numPositions++;
            continue;
        }

        // Otherwise add to stack.
        for (int i = COLS - 1; i > -1; i--)
        {
            if (canPlay(&current, i))
            {
                Position newPosition;
                copyPosition(&newPosition, &current);
                play(&newPosition, i);
                stack.push(newPosition);
            }
        }
    }

    // std::cout << "Found " << numPositions << " depth " << S << " positions" << std::endl;
    // std::cout << "Found " << numLeaves << " depth " << S << " leaf nodes" << std::endl;

    int *d_scores;
    Position *d_positions;
    PositionEntry *d_hashMap;

    if (numPositions > 0)
    {
        cudaMalloc(&d_positions, sizeof(Position) * 7 * 7 * 7 * 7 * 7 * 7 * 7);
        cudaMalloc(&d_scores, sizeof(int) * 7 * 7 * 7 * 7 * 7 * 7 * 7);
        cudaMalloc(&d_hashMap, sizeof(PositionEntry) * TABLE_SIZE);

        cudaMemcpy(d_positions, &h_positions, sizeof(Position) * 7 * 7 * 7 * 7 * 7 * 7 * 7, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scores, &h_scores, sizeof(int) * 7 * 7 * 7 * 7 * 7 * 7 * 7, cudaMemcpyHostToDevice);

        int numThreads = std::min(numPositions, 256);
        dim3 DimBlock(numThreads);
        dim3 DimGrid((numPositions + numThreads - 1) / numThreads);

        parallelGTS<<<DimGrid, DimBlock>>>(d_positions, d_scores, numPositions, d_hashMap);

        if (cudaMemcpy(&h_scores, d_scores, sizeof(int) * 7 * 7 * 7 * 7 * 7 * 7 * 7, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cout << "Error copying scores." << std::endl;
        }

        int max = -23;
        int indexOfReturn = 0;
        for (int i = 0; i < numPositions; i++)
        {
            scoreTable[h_positions[i]] = h_scores[i];
        }
        std::cout << std::endl;
        // }
    }
    int numLeaf = 0;
    score = finalNegamax(&initialPosition, -(COLS * ROWS) / 2, (COLS * ROWS) / 2, scoreTable, 0);
    std::cout << "Score: " << score << std::endl;
}