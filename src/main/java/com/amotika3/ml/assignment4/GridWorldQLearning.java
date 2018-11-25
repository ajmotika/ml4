package com.amotika3.ml.assignment4;

import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.List;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;

public class GridWorldQLearning {

	GridWorldDomain gwdg;
	OOSADomain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	
	public static void main(String[] args) {
		GridWorldQLearning example = new GridWorldQLearning();
		String outputPath = "output/"; //directory to record results
		
		example.runQLearningIteration(outputPath);
	}
	
	public GridWorldQLearning(){
		gwdg = new GridWorldDomain(11, 11);
		gwdg.setMapToFourRooms();
		tf = new GridWorldTerminalFunction(10, 10);
		gwdg.setTf(tf);
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();
		
		initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
		
		VisualActionObserver observer = new VisualActionObserver(domain, 
				GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		observer.initGUI();
		env.addObservers(observer);	
	}
	
	public void runQLearningIteration(String outputPath){
		List<Long> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Integer> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		Episode e = null;
		for (int i = 0; i < 100; i++) {
			long startTime = System.currentTimeMillis();
			
			
			LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, .99, .99);

			//run learning for 50 episodes
			for(int j = 0; j < 100; i++){
				e = agent.runLearningEpisode(env);
				env.resetEnvironment();
			}
			
			runtime.add(System.currentTimeMillis() - startTime);
			reward.add(this.calcRewardInEpisode(e));
			steps.add(e.numTimeSteps());
		}
		
		policyFunctionVisualization((ValueFunction)planner, p);
	}
	
	public double calcRewardInEpisode(Episode ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}
	
	public void policyFunctionVisualization(ValueFunction valueFunction, Policy p){
		List<State> allStates = StateReachability.getReachableStates(initialState, 
				domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, 11, 11, valueFunction, p);
		gui.initGUI();
	}
	
}
