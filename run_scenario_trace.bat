cd jar
java -jar JADEPlanner.jar jason-FullPlannerPlugin.log -log4j C:\Users\oracle\Anaconda3\github\Planner\jar\log4j-trace.props -s
call python create_log_for_analysis.py
copy /y C:\Users\oracle\Anaconda3\github\Planner\jar\log_for_analysis.log C:\Users\oracle\Anaconda3\github\Planner\jar\input\log_for_analysis.log
copy /y C:\Users\oracle\Anaconda3\github\Planner\jar\log_for_analysis.log C:\Users\oracle\Anaconda3\github\Planner\input\log_for_analysis.log
cd ..
call python read.py log_for