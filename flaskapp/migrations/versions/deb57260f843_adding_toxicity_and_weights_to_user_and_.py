"""adding toxicity and weights to user and post model

Revision ID: deb57260f843
Revises: dd8a07d3cad9
Create Date: 2020-12-01 22:08:41.187095

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'deb57260f843'
down_revision = 'dd8a07d3cad9'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('toxicity')
    op.drop_table('weights')
    op.add_column('post', sa.Column('flirtation', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('identity_attack', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('insult', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('profanity', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('sexual_explicit', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('threat', sa.Float(), nullable=True))
    op.add_column('post', sa.Column('toxicity', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('flirtation', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('identity_attack', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('insult', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('profanity', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('sexual_explicit', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('threat', sa.Float(), nullable=True))
    op.add_column('user', sa.Column('toxicity', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'toxicity')
    op.drop_column('user', 'threat')
    op.drop_column('user', 'sexual_explicit')
    op.drop_column('user', 'profanity')
    op.drop_column('user', 'insult')
    op.drop_column('user', 'identity_attack')
    op.drop_column('user', 'flirtation')
    op.drop_column('post', 'toxicity')
    op.drop_column('post', 'threat')
    op.drop_column('post', 'sexual_explicit')
    op.drop_column('post', 'profanity')
    op.drop_column('post', 'insult')
    op.drop_column('post', 'identity_attack')
    op.drop_column('post', 'flirtation')
    op.create_table('weights',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.Column('toxicity', sa.FLOAT(), nullable=True),
    sa.Column('threat', sa.FLOAT(), nullable=True),
    sa.Column('sexual_explicit', sa.FLOAT(), nullable=True),
    sa.Column('profanity', sa.FLOAT(), nullable=True),
    sa.Column('insult', sa.FLOAT(), nullable=True),
    sa.Column('identity_attack', sa.FLOAT(), nullable=True),
    sa.Column('flirtation', sa.FLOAT(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('toxicity',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('post_id', sa.INTEGER(), nullable=False),
    sa.Column('toxicity', sa.FLOAT(), nullable=True),
    sa.Column('threat', sa.FLOAT(), nullable=True),
    sa.Column('sexual_explicit', sa.FLOAT(), nullable=True),
    sa.Column('profanity', sa.FLOAT(), nullable=True),
    sa.Column('insult', sa.FLOAT(), nullable=True),
    sa.Column('identity_attack', sa.FLOAT(), nullable=True),
    sa.Column('flirtation', sa.FLOAT(), nullable=True),
    sa.ForeignKeyConstraint(['post_id'], ['post.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###