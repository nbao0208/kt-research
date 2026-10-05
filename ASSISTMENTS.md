# An Explanation on How to Interpret our Data Sets

Many of the log files you will get will be from the "problem-logs" table where each row represents a single problem of each student. There will be extra information like the number of hints used for the problem or the time it took for a student to make their first response (called "first_response_time" in the table below). Problem_log_id refers to the unique row in the table base. We will sometime "join" into this file other stuff that is at the student or assignment level, so those would be duplicated (user id and sequences id would be duplicated).

**Professor [Neil Heffernan](https://www.neilheffernan.net/home) has documented with videos what many of these fields mean. If you run your own experiment, there is a second set of descriptions of some of these fields in the files we give out that could be helpful.**

## Column Headings

* **user_id**
* The ID of the student doing the problem.



We have a set of features that tell researchers something about how students did prior to starting the problem set of interest.

* **prior_problem_count**
* The number of problems the student had completed in ASSISTments prior to this assignment. This allows you to know something about the student before they did this problem set.  You could use this to split student into low and high incoming ability to see in your intervention is effective.  If this is empty it means this student has not done any prior assistment work.


* **prior_correct**
* Goes with prior_problem_count. The number of problems the student had answered correctly in ASSISTments prior to this assignment.


* **prior_percent_correct**
* prior_correct/prior_problem_count. The percent of past ASSISTments problems the student got correct.



We have some features that are related to the problem set overall like how many problems they finished to if they were doing it for homework.

* **problem_count**
* Number of problems done by the student in this assignment.


* **assignment_started_count**
* The number of students who have started this assignment for the given class (assignment_id)


* **assignment_finished_count**
* The number of students who have finished this assignment for the given class (assignment_id)


* **assignment_homework_count**
* The number of students who have finished the assignment NOT between the hours of 7-3 server time.  This is considered school time.


* **homework_percent**
* The percent of students who did the assignment as homework (assignment_homework_count / assignment_finished_count)


* **problem_logs_id**
* Each problem the student does is recorded as a problem log. This is the ID of a problem log.   Problem_log is the table that the biggest in ASSISTments. About 10 millions problems were solved in 2012, so there are 10 million rows in the database. For each problem, there might be a few attempts that a child made and a few hint requests. We call them actions. We don't store actions in their own table but if we did it would be bigger as every problem has at least one action (if it is correct) and more if the student was incorrect. We store in the "actions" field of problem_log the actions with time stamps.


* **problem_logs_assignment_id**
* The ID of the assignment the problem log belongs to.


* **problem_logs_user_id**
* The same as user_id. It is the user id of the student whose problem log this is.


* **condition**
* We currently do not give condition that is coming soon.



**We have three new columns related to ARRS.**

* **ARRS Correctness.**
* The correctness on the first reassessment test.  A '1' represents that the student answered the question correctly and a '0' represents that the student answered the question incorrectly.  All other values (nulls or dashes) mean the student was never assigned an ARRS test or has not yet attempted the ARRS question.


* **ARRS Delay Days.**
* The number of days between when the student finished the skill builder and when the ARRS test was assigned.


* **ARRS Adaptive Mode.**
* ASSISTments now has an adaptive version of ARRS.  Students that take a longer number of items to learn, get reassessed earlier.  It is important when looking at data to know if the student was assigned in adaptive mode.  There is a nicely published paper on this features [here](https://drive.google.com/open?id=0B2X0QD6q79ZJRUVpOHljb1dNWlk&authuser=0).


* **guessed_gender**
* Detail on how we "guess" gender are listed [here](https://www.google.com/search?q=/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets/gender%3Fauthuser%3D0).  This with either be "Male", "Female" or "Unknown"


* **guessed_gender_2**
* Similar method like "guessed_gender", but has wider coverage as detailed [here](https://www.google.com/search?q=/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets/gender%3Fauthuser%3D0).  This with either be "Male", "Female" or "Unknown"


* **assistment_id**
* Similar to problem_id. The ID of a problem one will see in the builder. If a problem has multiple main problems and/or scaffolding, everything relating to one problem is called an assistment and has the same assistment_id. If you see problem logs with the same assistment number, they are multiple main problems(or scaffolding problems) that are part of the same overarching problem.


* **problem_id**
* The ID of the problem. If a problem has multiple main problems, each multiple main problem will have a different problem_id.


* **original**
* 1 = Main problem
* 0 = Scaffolding problem
* If a problem has scaffolding and the student answers incorrectly or asks for the problem to be broken into steps, a new problem will be created called a scaffolding problem. This creates a separate problem log row in the file with the variable original = 0.


* **correct**
* 1 = Correct on first attempt
* 0 = Incorrect on first attempt, or asked for help
* This column is often the target for prediction. (Minor note: Neil Heffernan notes that while this is true most of the time, we also have Essay questions that teachers can grade.  Neil thinks that if this value is say .25 that means the teacher gave it a 1 our of 4.)


* **answer_id**
* Only exists for multiple choice or choose all that apply questions
* A number =  the answer the student put in corresponds with one of the answers for that problem
* 0 or empty = the student put an answer not corresponding with one of the answers for that problem


* **answer_text**
* The answer the student entered. Or the value the student selected in a multiple choice or choose all that apply problem.


* **first_action**
* 0 = attempt
* 1 = hint
* 2 = scaffolding
* empty = student clicked on the problem but did nothing else


* **hint_count**
* Number of hints a student asked for during the duration of the problem


* **bottom_hint**
* 1 = The student asked for the bottom out hint
* 0 = The student did not ask for the bottom out hint.
* If this is blank it means the student did not ask for a hint.  Remember that for scaffolding questions they can not get a hint.



o   The bottom out hint is the last hint for a problem and will generally contain the problem’s answer.

* **attempt_count**
* Number of attempts(number of times a student entered an answer)


* **problem_start_time**
* Time the student started the problem


* **problem_end_time**
* Should be the time the student finished the problem, currently it is the time in milliseconds after the problem_start_time.


* **first_response_time**
* Time between start time and first student action(asking for hint or entering an answer) (in milliseconds)


* **overlap_time**
* The time in milliseconds for the student to complete the problem.  Ideally this is meant to be time it took the student to finish the problem.  For instance, if a student spent 10 seconds reading the problem, asked for a hint and spent 3 seconds reading it, and then spent 14 seconds typing in the correct answer, the overlap time would be 27 seconds (expressed in milliseconds.)
* This field is often computed incorrectly. Many data sets display overlap time the same as the first response time.  You could compute overlap time using other fields, like using the state time of two problems.


* **tutor_strategy_id**
* The types of tutoring strategies in ASSISTments are hints or scaffolding
* For a problem, the entire hint set or scaffolding set has a number, if a student used a tutoring strategy the number of that strategy is tutor_strategy_id.
* If there are multiple main problems the tutoring strategies for each main problem has a unique number.


* **assignment_type**
* Determined by assignment type id. Usually ClassAssignment, but sometimes ARRS or remedial.


* **teacher_comment**
* Currently unsure what this column refers to


* **network_state**
* The state of the network when the student did the assignment.
* CONNECTED = the student did the assignment while online
* DISCONNECTED = the student did the assignment while offline


* **assignment_logs_id**
* Each assignment the student does is recorded as an assignment log. This is the id of an assignment log.


* **assignment_logs_assignment_id**
* This is the id of the assignment. This number is linked to the assignment and is the same for all students that do this assignment.


* **assignment_logs_user_id**
* The same as user_id


* **assignment_start_time**
* same as problem_start_time


* **assignment_end_time**
* same as problem_end_time


* **instance**
* Currently unsure what this column refers to


* **assignment_logs_sequence**
* Currently unsure what this column refers to


* **effort_score**
* Currently unsure what this column refers to


* **assignment_logs_variables**
* If a variablized template, this will return any variables present.


* **last_worked_on**
* This is the last time the student worked on something from the assignment that goes with this problem log
* Only a date, not a time
* Used in seeing if the daily limit is reached


* **mastery_status**
* Only significant for a skill builder
* mastered = Student completed the number of problems required for mastery
* limit exceeded = Student exceeded the daily limit of problems for that skill builder
* not mastered exhausted =  Student attempted all the problems in that skill builder
* blank = Student did not fit into one of the previous three categories


* **assignment_logs_assignment_type**
* same as assignment_type


* **status_id**
* Used to tell if a teacher has marked an assignment as excused
* 10 = this assignment has been excused (for this student)


* **email_notification**
* Not used anymore


* **user_details_grade**
* This is the current grade of the student
* There appear to be students assigned to K(Kindergarten) that are in a higher grade


* **preferences**
* Old, not used anymore


* **registration_code**
* Not important for research, used for registration


* **email_validation_code**
* Not important for research, used for registration


* **heard_of_us_option_id**
* Not important for research, used for registration


* **heard_of_us_other_text**
* Not important for research, used for registration


* **class_assignments_id**
* Same as assignment_id


* **class_assignments_sequence_id**
* ASSISTments is very confusing in how they use "Problem Set" and "Sequence".  The same object that is called a "Sequence" in the database is exposed to teachers as a "Problem Set".


* **student_class_id**
* The ID of the class, the same for all students in the same class.  If you plan to use hierarchical linear modeling, you can use this for the class ID.  We can also give you a teacher ID.  You might also want to look at section ID (if its not in here we can give it to you: Korinn).


* **class_assignments_name**
* Currently unsure what this column refers to


* **points_possible**
* This is the points possible we have assigned to the problem. It is not significant now, but might be used in the future for computing partial credit.


* **has_files**
* This is very old. Currently unsure what this column refers to


* **due_date**
* Due date for the assignment. Can be set by the teacher as a date and time or just a date.


* **scope_type**
* Currently unsure what this column refers to


* **scope_id**
* Currently unsure what this column refers to


* **release_date**
* The time when the assignment showed up in student view


* **curriculum_item_id**
* Mainly used for folders. Aside from that, currently unsure what this column refers to.


* **sprial_assignment_id**
* Used to be used for ARRS, not useful anymore.


* **category**
* Currently unsure what this column refers to


* **class_assignments_assignment_type_id**
* Number corresponding with assignment type
* 1 = ClassAssignment
* Other numbers correspond with other assignment types (One of these means invdivudlzied assignment , one means reassessments  means ret
* 6 mean releasing


* **assigned_date**
* Should be the date the teacher assigned the assignment, in some problem logs the day appears to be missing and only the time is shown.


* **time_to_due_date**
* Currently unsure what this column refers to. Possibly the amount of time between when the student finished the assignment and when the assignment was due.


* **assignment_types_id**
* Same as class_assignments_assignment_type_id


* **assignment_types_assignment_type**
* Type of assignment, usually class, but probably can also be ARRS


* **assignment_types_origin**
* Who assigned the assignment. Can be Teacher, ARRS or placements


* **assignment_types_display_name**
* How the assignment type is displayed. The same as assignment_types_assignment_type.


* **user_roles_id**
* Each student has a distinct user_role_id that points to a place in the user roles table that tells what role they have.


* **role_id**
* 4 = Student
* 2 = Teacher
* 14 = Researcher


* **user_roles_user_id**
* Same as user_id


* **location_id**
* The id number for the school


* **type**
* For students this will always be school


* **location_type**
* For students this will always be school


* **school_id**
* ID of the school where the problem was assigned.


* **district_id**
* The id of the district


* **verified**
* Not useful for researchers. Students are only allowed to join verified schools (where verified = TRUE)


* **locked**
* Currently unsure what this column refers to


* **districts_id**
* Id number for the student’s district


* **district_name**
* Name of the student’s district


* **districts_state**
* Student’s state


* **districts_state_id**
* Id for the student’s state


* **districts_verfied**
* Not useful for researchers. Teachers are only allowed to join verified districts(where districts_verified = TRUE).


* **state_id**
* The id of the student’s state


* **state_name**
* Name of the student’s state


* **abbreviation**
* Abbreviation of the student’s state


* **state_locked**
* Not useful for researchers. Teachers are only allowed to join verified states(where state_locked = TRUE)


* **assistment_id**
* described above


* **assistments_name**
* The name of the assistment, will be cut off if too long


* **description**
* Is quite old, and doesn't appear to be used


* **assistments_created_at**
* Time the assistment was created. Should be a date and time, but sometimes is showing up as only a time.


* **assistments_position**
* Should be blank


* **assistment_type_id**
* 1 = normal problem
* 2 = variablized template


* **sequence_id**
* same as class_assignments_sequence_id


* **head_section_id**
* Not important for research. The head section decides if a problem is linear or a skill builder.


* **sequences_created_at**
* Should be the date the sequence was created, but currently doesn't appear to be working.


* **sequence_name**
* The title of the problem set (problem set and sequence are the same thing)


* **sequence_description**
* ASSISTments does not allow the sequence description to be changed anymore. If you are looking at an old problem set, there could be a sequence description.


* **quality_level_id**
* 1 = uncertified
* 3 = WPI certified


* **sequences_parameters**
* Parameters returned from the quick builder


* **sequences_updated_at**
* The last date the sequence was updated



## Column Headings Not Currently Used

* **order_id**
* These id's are chronological, and refer to the id of the original problem log.


* **tutor_mode**
* tutor, test mode, pre-test, or post-test


* **problem_set_type**
* Linear - Student completes all problems in a predetermined order.
* Random - Student completes all problems, but each student is presented with the problems in a different random order.
* Mastery - Random order, and student must "master" the problem set by getting a certain number of questions correct in a row before being able to continue.


* **base_sequence_id**
* This is to account for if a sequence has been copied. This will point to the original copy, or be the same as sequence_id if it hasn't been copied.


* **skill_id**
* ID of the skill associated with the problem.
* For the skill builder dataset, different skills for the same data record are in different rows. This means if a student answers a multi skill question, this record is duplicated several times, and each duplication is tagged with one of the multi skills.
* For the non skill builder dataset, different skills for the same data record are in the same row, separated with comma.


* **skill_name**
* Skill name associated with the problem.
* For the skill builder dataset, different skills for the same data record are in different rows. This means if a student answers a multi skill question, this record is duplicated several times, and each duplication is tagged with one of the multi skills.
* For the non skill builder dataset, different skills for the same data record are in the same row, separated with comma.


* **hint_total**
* Number of possible hints on this problem.  We tell you the total number of hints so you can compute something like a % of hints used.  Not all problems have all the same number of hints.


* **template_id**
* The template ID of the ASSISTment. ASSISTments with the same template ID have similar questions.


* **opportunity**
* The number of opportunities the student has to practice on this skill.
* For the skill builder dataset, opportunities for different skills of the same data record are in different rows. This means if a student answers a multi skill question, this record is duplicated several times, and each duplication is tagged with one of the multi skills and the corresponding opportunity count.
* For the non skill builder dataset, opportunities for different skills of the same data record are in the same row, separated with comma.


* **opportunity_original**
* The number of opportunities the student has to practice on this skill counting only original problems.
* For the skill builder dataset, original opportunities for different skills of the same data record are in different rows. This means if a student answers a multi skill question, this record is duplicated several times, and each duplication is tagged with one of the multi skills and the corresponding original opportunity count.
* For the non skill builder dataset, original opportunities for different skills of the same data record are in the same row, separated with comma.