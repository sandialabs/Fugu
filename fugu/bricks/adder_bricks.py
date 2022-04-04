class streaming_adder(Brick):
    
    """
    streaming adder function. 
    Brad Aimone
    jbaimon@sandia.gov
    
    """
    
    def __init__(self, name=None):
        super().__init__()
        self.is_built = False
        self.dimensionality = {'D': 2}
        self.name = name
        self.supported_codings = ['binary-L']
        
    def build(self, graph, dimensionality, control_nodes, input_lists, input_codings):
        """
        Build streaming adder brick. 

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        
        if len(input_codings) != 2:
            raise ValueError("adder takes in 2 input on size n")
            
        output_codings = [input_codings[0]]
        
        new_complete_node_name = self.name + '_complete'
        new_begin_node_name = self.name + '_begin'
        
        graph.add_node(new_begin_node_name, index = -2, threshold = 0.0, decay =0.0, p=1.0, potential=0.0)
        graph.add_node(new_complete_node_name,
                      index = -1,
                      threshold = .9,
                      decay =0.0,
                      p=1.0,
                      potential=0.0)
        

        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name, weight=0.5, delay=3)
        graph.add_edge(control_nodes[1]['complete'], new_complete_node_name, weight=0.5, delay=3)

        graph.add_edge(control_nodes[0]['begin'], new_begin_node_name, weight=1.0, delay=2)

        complete_node = new_complete_node_name
        begin_node = new_begin_node_name
        
     
        l = len(input_lists[0])
        
        #nodes
        graph.add_node(self.name + 'add', threshold=.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(self.name + 'carry0', threshold=1.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(self.name + 'carry1', threshold=2.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(self.name + 'out', threshold=.9, decay=1.0, p=1.0, potential=0.0)
        #edges
        graph.add_edge(input_lists[0][0], self.name + 'add', weight=1.0, delay=1)
        graph.add_edge(input_lists[1][0], self.name + 'add', weight=1.0, delay=1)
        graph.add_edge(input_lists[0][0], self.name + 'carry0', weight=1.0, delay=1)
        graph.add_edge(input_lists[1][0], self.name + 'carry0', weight=1.0, delay=1)
        graph.add_edge(input_lists[0][0], self.name + 'carry1', weight=1.0, delay=1)
        graph.add_edge(input_lists[1][0], self.name + 'carry1', weight=1.0, delay=1)
        
        graph.add_edge(self.name + 'carry0', self.name + 'add', weight=1.0, delay=1)
        graph.add_edge(self.name + 'carry0', self.name + 'carry0', weight=1.0, delay=1)
        graph.add_edge(self.name + 'carry0', self.name + 'carry1', weight=1.0, delay=1)
        
        graph.add_edge(self.name + 'add', self.name + 'out', weight=1.0, delay=1)
        graph.add_edge(self.name + 'carry0', self.name + 'out', weight=-1.0, delay=1)
        graph.add_edge(self.name + 'carry1', self.name + 'out', weight=1.0, delay=1)
        
        self.is_built=True
        
        output_lists = [[self.name + 'out']]
        
        return (graph, self.dimensionality, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)
    
class temporal_shift(Brick):
    
    """
    temporal shift function. 
    Brad Aimone
    jbaimon@sandia.gov
    
    """
    
    def __init__(self, name=None, shift_length = 1):
        super().__init__()
        self.is_built = False
        self.dimensionality = {'D': 2}
        self.name = name
        self.supported_codings = ['binary-L']
        self.shift_length = shift_length
        
    def build(self, graph, dimensionality, control_nodes, input_lists, input_codings):
        """
        Build bit shift brick. 

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        
        if len(input_codings) != 1:
            raise ValueError("bit_shift takes in 1 input on size n")
            
        output_codings = [input_codings[0]]
        
        new_complete_node_name = self.name + '_complete'
        new_begin_node_name = self.name + '_begin'
        
        graph.add_node(new_begin_node_name, index = -2, threshold = 0.0, decay =0.0, p=1.0, potential=0.0)
        graph.add_node(new_complete_node_name,
                      index = -1,
                      threshold = 0.0,
                      decay =0.0,
                      p=1.0,
                      potential=0.0)
        

        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name, weight=1.0, delay=self.shift_length)
        graph.add_edge(control_nodes[0]['begin'], new_begin_node_name, weight=1.0, delay=1)

        complete_node = new_complete_node_name
        begin_node = new_begin_node_name
        
        l = len(input_lists[0])
        
        #nodes
        graph.add_node(self.name + 'out', threshold=.9, decay=1.0, p=1.0, potential=0.0)
        #edges
        graph.add_edge(input_lists[0][0], self.name + 'out', weight=1.0, delay=self.shift_length)
        
        self.is_built=True
        
        output_lists = [[self.name + 'out']]
        
        return (graph, self.dimensionality, [{'complete': complete_node, 'begin':begin_node}], output_lists, output_codings)
    
    
class streaming_scalar_multiplier(Brick):
    
    """
    streaming scalar multiplier function. 
    Brad Aimone
    jbaimon@sandia.gov
    
    """
    
    def __init__(self, name=None, shift_length = 1):
        super().__init__()
        self.is_built = False
        self.dimensionality = {'D': 2}
        self.name = name
        self.supported_codings = ['binary-L']
        self.shift_length = shift_length
        
    def build(self, graph, dimensionality, control_nodes, input_lists, input_codings, alpha = .125):
    #def streaming_scalar_multiplier(alpha = .125):
        # Scalar multiplication example

        scaffold = Scaffold()

        #alpha = .3125
        bit_length = 8
        # Convert alpha to little-endian binary
        a1=bin(int(alpha*2**bit_length))[2:].zfill(bit_length)
        a1=a1[::-1]
        print(a1)


        # For scalar multiplication, what we want to do is create a Scaffold taking the original input, x, and a zeros input
        # 
        # This is inefficient, but we will bit shift x to all possible stages.  But we will use the zeros input for each adder who place entry in the scalar is zero

        scaffold.add_brick(Vector_Input(np.array([[0, 1, 1, 0, 1, 0, 0, 0]]), coding='binary-L', name='Input0', time_dimension = True), 'input' )
        scaffold.add_brick(Vector_Input(np.array([[0, 0, 0, 0, 0, 0, 0, 0]]), coding='binary-L', name='Input_zeros', time_dimension = True), 'input' )

        for i in range(1, bit_length):
            # Create a bit shifted version of 'x' for each of the place values of alpha
            scaffold.add_brick(temporal_shift(name='shift'+str(i)+'_', shift_length=1+3*(i-1)), [(0,0)], output=True)    

        for i in range(1,bit_length):
            if(i==1):
                # First adder potentially takes in 'x' and 'x<<1'  Need to check if both are used; otherwise pass in zeros
                if(a1[i-1]=='1'):
                    in_1=0
                else:
                    in_1=1 # First place of scalar is empty, so pass in zeros
            else:
                # All other adders use previous adder as an input
                in_1=bit_length-1+i # 7+i
            if(a1[i]=='1'):
                # For each adder, check if element = 1, otherwise pass in zero as input
                in_2=1+i
            else:
                in_2=1

            #print(i, in_1, in_2)
            scaffold.add_brick(streaming_adder(name='adder_'+str(i)+'_'), [(in_1,0), (in_2, 0)], output=True)


        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        return scaffold