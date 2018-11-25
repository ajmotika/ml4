package com.amotika3.ml.assignment4;

import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction.IntPair;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;

public class GridWorldSuboptimalTeminals {

	GridWorldDomain gwdg;
	OOSADomain domain;
	RewardFunction rf;
	GridWorldTerminalFunction tf;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	Map<Integer, List<Double>> runtimeMap = new HashMap<>();
	Map<Integer, List<Double>> rewardMap = new HashMap<>();
	Map<Integer, List<Double>> stepMap = new HashMap<>();
	
	protected static int[][] userMap = new int[][] { 
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{ 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0},
		{ 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
		{ 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1},
		{ 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},	
		{ 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0},
		{ 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0},
		{ 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0},
		{ 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1},
		{ 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0},
		{ 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0},
		{ 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0},
		{ 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},
		{ 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0},
		{ 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		{ 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0},
		{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{ 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
	};
	
	public static void main(String[] args) {
		if (args.length < 2) {
			throw new RuntimeException("Input needs two args: probabilityOfSuccess and discountFactor");
		}
			
		double probabilityOfSuccess = Double.parseDouble(args[0]);
		double discountFactor = Double.parseDouble(args[1]);
		GridWorldSuboptimalTeminals valueIteration = new GridWorldSuboptimalTeminals(probabilityOfSuccess);
		GridWorldSuboptimalTeminals policyIteration = new GridWorldSuboptimalTeminals(probabilityOfSuccess);
		GridWorldSuboptimalTeminals qLearning = new GridWorldSuboptimalTeminals(probabilityOfSuccess);
		String outputPath = "output/"; //directory to record results
		
		for (int i = 5; i <= 1000; i += 10) {
			valueIteration.runValueIteration(outputPath, 500, discountFactor);
			policyIteration.runPolicyIteration(outputPath, 500, discountFactor);	
		}
	
		qLearning.runQLearningIteration(outputPath, .99);
		
		valueIteration.printRuns("Value Iteration");
		policyIteration.printRuns("Policy Iteration");
	}
	
	class BasicRewardFunction implements RewardFunction {

		protected StateConditionTest gc;

		public BasicRewardFunction(StateConditionTest gc) {
			this.gc = gc;
		}

		@Override
		public double reward(State s, Action a, State sprime) {
			// are they at goal location?
			if (gc.satisfies(sprime)) {
				double xPos = ((int) sprime.get("agent:x")) + 1;
				double yPos = ((int) sprime.get("agent:y")) + 1;
				
				return xPos * yPos > 50 ? 100 : 5;
			}
			return -1;
		}
	}
	
	public GridWorldSuboptimalTeminals(double probSucceed){
		gwdg = new GridWorldDomain(18, 11);
		gwdg.setMap(userMap);
		
		tf = new GridWorldTerminalFunction();
		tf.markAsTerminalPosition(17, 10); // Optimal (100)
		tf.markAsTerminalPosition(0, 10);  // Suboptimal (5)
		tf.markAsTerminalPosition(17, 0);  // Suboptimal (5)
		
		rf = new BasicRewardFunction(new TFGoalCondition(tf));
		gwdg.setTf(tf);
		gwdg.setRf(rf);
		gwdg.setProbSucceedTransitionDynamics(probSucceed);
		
		domain = gwdg.generateDomain();
		
		initialState = new GridWorldState(new GridAgent(0, 0));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}
	
	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}
	
	public void runValueIteration(String outputPath, int numIterations, double discountFactor){
		List<Double> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Double> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		for (int i = 0; i < 100; i++) {
			long startTime = System.currentTimeMillis();
			
			planner = new ValueIteration(domain, discountFactor, hashingFactory, 0.1, numIterations);
			p = planner.planFromState(initialState);
			runtime.add(1.0 * System.currentTimeMillis() - startTime);
			
			Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
			
			reward.add(this.calcRewardInEpisode(episode));
			steps.add(1.0 * episode.numTimeSteps());
		}
		
		this.runtimeMap.put(numIterations, runtime);
		this.rewardMap.put(numIterations, reward);
		this.stepMap.put(numIterations, steps);
		
		if (numIterations == 50)
			valueFunctionVisualization((ValueFunction)planner, p);
	}
	
	public void runPolicyIteration(String outputPath, int numIterations, double discountFactor){
		List<Double> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Double> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		for (int i = 0; i < 100; i++) {
			long startTime = System.currentTimeMillis();
			
			planner = new PolicyIteration(domain, discountFactor, hashingFactory, 0.1, numIterations, numIterations);
			p = planner.planFromState(initialState);
			runtime.add(1.0 * System.currentTimeMillis() - startTime);
			
			Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
			
			reward.add(this.calcRewardInEpisode(episode));
			steps.add(1.0 * episode.numTimeSteps());
		}
		
		this.runtimeMap.put(numIterations, runtime);
		this.rewardMap.put(numIterations, reward);
		this.stepMap.put(numIterations, steps);
		
		if (numIterations == 50)
			valueFunctionVisualization((ValueFunction)planner, p);
	}
	
	public void runQLearningIteration(String outputPath, double discountFactor){
		List<Long> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Integer> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		Episode e = null;
//		for (int i = 0; i < 100; i++) {
		long startTime = System.currentTimeMillis();
		
		
		QLearning agent = new QLearning(domain, discountFactor, hashingFactory, .3, .1);

		//run learning for 50 episodes
		for(int j = 0; j < 100000; j++){
			e = agent.runLearningEpisode(env);
			env.resetEnvironment();
		}
		
		//ini(rf, tf, 1);
		p = agent.planFromState(initialState);
		
		runtime.add(System.currentTimeMillis() - startTime);
		reward.add(this.calcRewardInEpisode(e));
		steps.add(e.numTimeSteps());

		
		valueFunctionVisualization((ValueFunction)agent, p);
	}
	
	
	public void printRuns(String name) {
		System.out.println("---------------------------------");
		System.out.println("---------------------------------");
		System.out.println("----------" + name + "-----------");
		
		System.out.println("----------Runtimes-----------");
		for (int key : this.runtimeMap.keySet().stream().sorted().collect(Collectors.toList())) {
			System.out.println(key + "," + this.getStats(this.runtimeMap.get(key)));
		}
		
		System.out.println("----------Rewards-----------");
		for (int key : this.runtimeMap.keySet().stream().sorted().collect(Collectors.toList())) {
			System.out.println(key + "," + this.getStats(this.rewardMap.get(key)));
		}
		
		System.out.println("----------Steps-----------");
		for (int key : this.runtimeMap.keySet().stream().sorted().collect(Collectors.toList())) {
			System.out.println(key + "," + this.getStats(this.stepMap.get(key)));
		}
	}
	
	public String getStats(List<Double> vals) {
		double mean = vals.stream().mapToDouble(a -> (double) a).average().getAsDouble();
		
		double temp = vals.stream().mapToDouble(a -> (double) a).reduce((a,b) -> a + (b-mean) * (b-mean)).getAsDouble();
		
		double var = temp/(vals.size() - 1);
		
		return mean + ", " + Math.sqrt(var);
	}
	
	public double calcRewardInEpisode(Episode ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}
	
	public void valueFunctionVisualization(ValueFunction valueFunction, Policy p){
		List<State> allStates = StateReachability.getReachableStates(initialState, 
				domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, 18, 11, valueFunction, p);
		gui.initGUI();
	}
}
