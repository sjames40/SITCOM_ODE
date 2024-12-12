# SITCOM_phase-retrieval
To address the instability issue in SITCOM phase retrieval, we drew inspiration from DAPS and incorporated an ODE solver to replace the Tweedie formula update, enabling a more robust and accurate solution. The revised approach integrates an Euler solver, enhanced with a data consistency projection, ensuring the ODE solution maintains greater consistency throughout the process. Additionally, we implemented a data consistency update guided by a Lagrange multiplier, further refining the overall stability and performance of the method. The updated code is provided below.

