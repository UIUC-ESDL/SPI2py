import openmdao.api as om

class MyCheckComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('a', 1.0)
        self.add_input('b', 0.5)
        self.add_output('x', 0.0)
        # because we set method='cs' here, OpenMDAO automatically knows to allocate
        # complex nonlinear vectors
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        if self.under_complex_step:
            print('under complex step')
        else:
            print('not under complex step')

        # outputs['x'] = a * 2.
        outputs['x'] = (a**2+b**2)**(1/2)

p = om.Problem()
p.model.add_subsystem('comp', MyCheckComp())
# don't need to set force_alloc_complex=True here since we call declare_partials with
# method='cs' in our model.
p.setup()

# during run_model, our component's compute will *not* be running under complex step
p.run_model()

# during compute_partials, our component's compute *will* be running under complex step
J = p.compute_totals(of=['comp.x'], wrt=['comp.a','comp.b'])
print(J['comp.x', 'comp.a'], J['comp.x', 'comp.b'])