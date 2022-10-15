<div>
    <h1>Reinforcement Learning</h1>
    <p>This repository contains Python scripts for implementing various reinforcement learning algorithms, including REINFORCE and Actor-Critic. The scripts demonstrate the application of these algorithms on simple examples as well as the CartPole environment from the OpenAI Gym.</p>
    <h2>Files</h2>
    <ul>
        <li><code>policynet.py</code>: Implements a policy module for the REINFORCE algorithm.</li>
        <li><code>acnet.py</code>: Implements a policy module for the Actor-Critic algorithm.</li>
        <li><code>reinforce-cartpole.py</code>: Applies the REINFORCE algorithm to solve the CartPole environment from OpenAI Gym.</li>
    </ul>
    <h2>Usage</h2>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/ns9920/reinforcement-learning.git</code></pre>
        </li>
        <li>Navigate to the project directory:
            <pre><code>cd reinforcement-learning</code></pre>
        </li>
        <li>Install the required dependencies:
            <pre><code>pip install gymnasium torch numpy</code></pre>
        </li>
        <li>Run the desired script:
            <ul>
                <li>For the REINFORCE algorithm on a simple example:
                    <pre><code>python policynet.py</code></pre>
                </li>
                <li>For the Actor-Critic algorithm on a simple example:
                    <pre><code>python acnet.py</code></pre>
                </li>
                <li>For the REINFORCE algorithm on the CartPole environment:
                    <pre><code>python reinforce-cartpole.py</code></pre>
                </li>
            </ul>
        </li>
    </ol>
    <h2>Dependencies</h2>
    <p>This project requires the following Python packages:</p>
    <ul>
        <li>gymnasium</li>
        <li>torch</li>
        <li>numpy</li>
    </ul>
    <p>You can install these packages using the following command:</p>
    <pre><code>pip install gymnasium torch numpy</code></pre>
    <h2>Contributing</h2>
    <p>Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.</p>
</div>
