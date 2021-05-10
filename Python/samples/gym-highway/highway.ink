inkling "2.0"
using Number

type SimState{
    vehicle1: number[5],
    vehicle2: number[5],
    vehicle3: number[5],
    vehicle4: number[5],
    vehicle5: number[5],
    gym_reward: number,
    gym_terminal: Number.Bool
}

type GameState {
    vehicle1: number[5],
    vehicle2: number[5],
    vehicle3: number[5],
    vehicle4: number[5],
    vehicle5: number[5],
}

function Reward(State: SimState) {

    return State.gym_reward
}

function Terminal(State: SimState) {

    return State.gym_terminal
}

type CarSteer {
    steer: number<0..4 step 1>,
}

type HighwayConfig {
    controlled_vehicles: number,
    ego_spacing: number,
    vehicles_count: number,
    lanes_count: number
}

simulator HighwaySimulator(action: CarSteer, config: HighwayConfig): SimState {
}

graph (input: GameState): CarSteer {
    concept Drive(input): CarSteer {
        curriculum {
            source HighwaySimulator
            reward Reward
            terminal Terminal
            lesson TwoLanes {
                scenario {
                    vehicles_count: 15,
                    lanes_count: 2
                }
            }
            lesson MoreLanes {
                scenario {
                    vehicles_count: 15,
                    lanes_count: number<2..4>
                }
            }
            lesson MoreCars {
                scenario {
                    vehicles_count: number<15..40>,
                    lanes_count: number<2..4>
                }
            }
        }
    }
    output Drive
}
