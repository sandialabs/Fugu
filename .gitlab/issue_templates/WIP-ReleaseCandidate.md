## Release Candidate Info

- **Branching Point SHA**: TBD    
- **Source tag:**: `TBD`  
- **Destination tag:**: `TBD`  

-----
-----

## Workflow
### Stage: Backlog

#### Step 1: Preparation
- Identify branching point commit for this RC (choose one case)  
  - [ ] **Case**: This RC is the freeze commit or prior (including RC1) 
    - [ ] Choose a commit in `master` as the branching point
    - [ ] Tag branching point commit in `master` with RC prepare tag: vX.Y.0-rcZ-prepare (e.g. `v8.1.0-rc1-prepare`)
  - [ ] **Case**: This RC is after the feature freeze
    - [ ] Use the previous RC commit and tag in quality as the branching  point
- [ ] Paste RC branching point SHA into the [info](#release-candidate-info) section above
- [ ] Paste RC source tag into the [info](#release-candidate-info) section above
- [ ] Paste RC destination tag into the [info](#release-candidate-info) section above
- [ ] Ensure this issue is assigned to the release milestone
- [ ] Ensure ~RC and ~Release labels are applied to this issue
- [ ] Link this issue to the release meta-issue
- [ ] Link this issue to any previous RC issues (if applicable)

> **Development Board**: Move issue from Backlog to ~"Backlog".

-----

### Stage: ~"Backlog" 
- [ ] Assign to release manager(s)

> **Development Board**: Move issue from ~"Backlog" to ~"In\-Progress".

-----

### Stage: ~"In\-Progress" 
#### Step 1: RC MR and Tagging
- [ ] Create merge request and preparation branch
  - Use the GUI button here on the issue to "Create merge request"
  - Use the RC source tag noted above as the source for MR
  - Apply the "RC to Qual" MR template
  - The branch will be used to prepare any changes included for this RC
- [ ] Ensure the MR is marked WIP
- [ ] Ensure the MR is targeting `quality`

#### Step 2: Develop RC
- [ ] Cherry-pick required changes from `master` (e.g. security, critical bug fixes, etc.)  
- [ ] Ensure pipelines are green on preparation branch
- [ ] Resolve any discussions in MR
- [ ] Resolve WIP status in MR

> **Development Board**: Move issue from ~"In\-Progress"  to ~"In\-Review".

-----

### Stage: ~"In\-Review" 
- [ ] Merge to `quality` by accepting MR
- [ ] Tag in `quality` with RC version number vX.Y.0-rcZ (e.g. `v8.1.0-rc1`)
- [ ] Paste new RC tag into the [info](#release-candidate-info) section above
- [ ] Confirm RC has been deployed to `quality` environment
- [ ] Confirm pipelines are green in `quality`

> **Release Board**: Move issue from Open to ~"In\-Quality".  
> **Release Board**: Move previous RC issue from ~"In\-Quality" to Closed (if applicable).  



/label ~RC 
/label ~Release 

